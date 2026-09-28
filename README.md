# CamemBERTQA

Extractive question answering in French with [CamemBERT](https://camembert-model.fr/), fine-tuned on the [FQuAD](https://fquad.illuin.tech/) dataset.

This is an archived internship project from 2020. It explores:

- running a pretrained French QA model on short and long documents, and
- fine-tuning `camembert-base` with two custom heads between the encoder and the span-prediction layer: a Transformer encoder written from scratch, and a BiLSTM.

## Notebooks

| Notebook | What it does |
|---|---|
| [`01_pretrained_qa_inference`](notebooks/01_pretrained_qa_inference.ipynb) | Answers a question with [`fmikaelian/camembert-base-fquad`](https://huggingface.co/fmikaelian/camembert-base-fquad) and plots per-token start/end scores. |
| [`02_long_context_inference`](notebooks/02_long_context_inference.ipynb) | Handles contexts longer than CamemBERT's 512-token limit by splitting them into chunks and keeping the best-scoring answer. |
| [`03_finetune_transformer_head`](notebooks/03_finetune_transformer_head.ipynb) | Fine-tunes `camembert-base` + 2 Transformer encoder blocks on FQuAD. |
| [`04_finetune_bilstm_head`](notebooks/04_finetune_bilstm_head.ipynb) | Fine-tunes `camembert-base` + a BiLSTM on FQuAD. |

## Approach

Both fine-tuning notebooks share the same pipeline:

1. Flatten the SQuAD-format JSON into `(context, question, answer)` rows. Training contexts are limited to 500 characters.
2. Encode `question + context`, then locate the answer's start and end token positions.
3. Feed CamemBERT's hidden states through the custom head, then a linear layer that outputs start and end logits.
4. Train with the sum of the cross-entropy losses on the start and end positions (AdamW, constant schedule with warmup).
5. Score predictions on the validation set with Exact Match and F1, using the [FQuAD paper](https://arxiv.org/abs/2002.06071)'s adaptation of the SQuAD evaluation. Case, punctuation and French articles are ignored, and each prediction is compared with every gold answer.

## Results

No Exact Match / F1 scores are available yet: they require re-running the fine-tuning notebooks.

The 2020 run of the Transformer-head notebook (3 epochs on a Colab GPU) was scored with an earlier, more lenient metric that counted a prediction as correct if its span overlapped the gold answer at all. By that measure, 71.5% of the 3,188 FQuAD validation questions were answered correctly. That figure can't be compared with Exact Match / F1 or with published FQuAD results.

The 2020 BiLSTM run was not trained to completion, so no result is reported for it.

## Running the notebooks

The code targets the 2020 `transformers` 2.x API (`transformers.modeling_camembert`, tuple outputs), which later versions removed. Use the pinned versions:

```bash
python3.7 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

The fine-tuning notebooks need a GPU and the FQuAD JSON files (`train.json`, `valid.json`) in the working directory. The dataset is licensed separately and is not included in this repository. Request it from the [FQuAD website](https://fquad.illuin.tech/).

## License

The code is released under the [MIT License](LICENSE). FQuAD and the pretrained models are covered by their own licenses.
