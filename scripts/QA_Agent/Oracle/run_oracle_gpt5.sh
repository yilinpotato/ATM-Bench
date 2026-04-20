#!/usr/bin/env bash

set -euo pipefail

OPENAI_API_KEY="${OPENAI_API_KEY:-$(cat api_keys/.openai_key)}"
export OPENAI_API_KEY

# Optional OpenAI-compatible endpoint override (keeps default OpenAI endpoint if unset).
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
JUDGE_OPENAI_BASE_URL="${JUDGE_OPENAI_BASE_URL:-${OPENAI_BASE_URL}}"

MODEL_NAME="${MODEL_NAME:-gpt-4o-2024-08-06}"
# Comma-separated fallback list. First available model will be used.
MODEL_CANDIDATES="${MODEL_CANDIDATES:-${MODEL_NAME},gpt-4o-mini,gpt-4.1-mini}"
MODEL_TAG="gpt5"
ANSWER_REASONING_EFFORT="${ANSWER_REASONING_EFFORT:-medium}"
ANSWER_MAX_WORKERS="${ANSWER_MAX_WORKERS:-1}"
ANSWER_MAX_RETRIES="${ANSWER_MAX_RETRIES:-6}"
ANSWER_REQUEST_DELAY="${ANSWER_REQUEST_DELAY:-1.2}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-${MODEL_NAME}}"

if [[ "${ANSWER_REASONING_EFFORT}" == "none" ]]; then
  RUN_TAG="${MODEL_TAG}_no_reasoning_effort"
else
  RUN_TAG="${MODEL_TAG}_reasoning_${ANSWER_REASONING_EFFORT}"
fi

ATM_PREDICTIONS="output/QA_Agent/Oracle/${RUN_TAG}/atmbench/oracle_${RUN_TAG}.jsonl"
ATM_EVAL_DIR="output/QA_Agent/Oracle/${RUN_TAG}/atmbench/eval"
HARD_PREDICTIONS="output/QA_Agent/Oracle/${RUN_TAG}/hard/oracle_${RUN_TAG}.jsonl"
HARD_EVAL_DIR="output/QA_Agent/Oracle/${RUN_TAG}/hard/eval"

ORACLE_OPENAI_URL_ARGS=()
if [[ -n "${OPENAI_BASE_URL}" ]]; then
  ORACLE_OPENAI_URL_ARGS+=(--openai-base-url "${OPENAI_BASE_URL}")
fi

JUDGE_OPENAI_URL_ARGS=()
if [[ -n "${JUDGE_OPENAI_BASE_URL}" ]]; then
  JUDGE_OPENAI_URL_ARGS+=(--judge-openai-base-url "${JUDGE_OPENAI_BASE_URL}")
fi

IFS=',' read -r -a _model_candidates <<< "${MODEL_CANDIDATES}"
SELECTED_MODEL=""
for candidate in "${_model_candidates[@]}"; do
  candidate="${candidate//[[:space:]]/}"
  [[ -z "${candidate}" ]] && continue
  if OPENAI_BASE_URL="${OPENAI_BASE_URL}" OPENAI_API_KEY="${OPENAI_API_KEY}" CANDIDATE_MODEL="${candidate}" python - <<'PY'
import os
import sys

from openai import OpenAI

model = os.environ["CANDIDATE_MODEL"]
api_key = os.environ["OPENAI_API_KEY"]
base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None

client = OpenAI(api_key=api_key, base_url=base_url)
client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "ping"}],
    max_tokens=1,
    temperature=0,
)
print(model)
PY
  then
    SELECTED_MODEL="${candidate}"
    break
  fi
done

if [[ -z "${SELECTED_MODEL}" ]]; then
  echo "[oracle] ERROR: no available model in MODEL_CANDIDATES='${MODEL_CANDIDATES}' for current endpoint."
  echo "[oracle] Hint: set MODEL_NAME or MODEL_CANDIDATES to a model your provider exposes."
  exit 1
fi

if [[ "${SELECTED_MODEL}" != "${MODEL_NAME}" ]]; then
  echo "[oracle] MODEL_NAME '${MODEL_NAME}' unavailable, fallback to '${SELECTED_MODEL}'."
fi

MODEL_NAME="${SELECTED_MODEL}"
if [[ -z "${JUDGE_MODEL:-}" ]]; then
  JUDGE_MODEL="${MODEL_NAME}"
fi

python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "./data/atm-bench/atm-bench.json" \
  --media-source raw \
  --image-batch-results "./output/image/qwen3vl2b/batch_results.json" \
  --video-batch-results "./output/video/qwen3vl2b/batch_results.json" \
  --image-root "./data/raw_memory/image" \
  --video-root "./data/raw_memory/video" \
  --email-file "./data/raw_memory/email/emails.json" \
  --provider openai \
  --model "${MODEL_NAME}" \
  "${ORACLE_OPENAI_URL_ARGS[@]}" \
  --reasoning-effort "${ANSWER_REASONING_EFFORT}" \
  --max-workers "${ANSWER_MAX_WORKERS}" \
  --max-retries "${ANSWER_MAX_RETRIES}" \
  --request-delay "${ANSWER_REQUEST_DELAY}" \
  --timeout 120 \
  --output-file "${ATM_PREDICTIONS}"

if [[ ! -f "${ATM_PREDICTIONS}" ]]; then
  echo "[oracle] ERROR: prediction file missing: ${ATM_PREDICTIONS}"
  exit 1
fi

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench.json" \
  --predictions "${ATM_PREDICTIONS}" \
  --output-dir "${ATM_EVAL_DIR}" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model "${JUDGE_MODEL}" \
  "${JUDGE_OPENAI_URL_ARGS[@]}" \
  --judge-reasoning-effort minimal \
  --max-workers "${EVAL_MAX_WORKERS}"

python memqa/qa_agent_baselines/oracle/oracle_baseline.py \
  --qa-file "./data/atm-bench/atm-bench-hard.json" \
  --media-source raw \
  --image-batch-results "./output/image/qwen3vl2b/batch_results.json" \
  --video-batch-results "./output/video/qwen3vl2b/batch_results.json" \
  --image-root "./data/raw_memory/image" \
  --video-root "./data/raw_memory/video" \
  --email-file "./data/raw_memory/email/emails.json" \
  --provider openai \
  --model "${MODEL_NAME}" \
  "${ORACLE_OPENAI_URL_ARGS[@]}" \
  --reasoning-effort "${ANSWER_REASONING_EFFORT}" \
  --max-workers "${ANSWER_MAX_WORKERS}" \
  --max-retries "${ANSWER_MAX_RETRIES}" \
  --request-delay "${ANSWER_REQUEST_DELAY}" \
  --timeout 120 \
  --output-file "${HARD_PREDICTIONS}"

if [[ ! -f "${HARD_PREDICTIONS}" ]]; then
  echo "[oracle] ERROR: prediction file missing: ${HARD_PREDICTIONS}"
  exit 1
fi

python memqa/utils/evaluator/evaluate_qa.py \
  --ground-truth "./data/atm-bench/atm-bench-hard.json" \
  --predictions "${HARD_PREDICTIONS}" \
  --output-dir "${HARD_EVAL_DIR}" \
  --metrics em atm \
  --judge-provider openai \
  --judge-model "${JUDGE_MODEL}" \
  "${JUDGE_OPENAI_URL_ARGS[@]}" \
  --judge-reasoning-effort minimal \
  --max-workers "${EVAL_MAX_WORKERS}"
