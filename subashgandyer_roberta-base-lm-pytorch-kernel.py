import os

print("Files contained in the roberta-base dataset")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import tokenizers



ROBERTA_PATH = "../input/roberta-base"

TOKENIZER = tokenizers.ByteLevelBPETokenizer(

    vocab_file=f"{ROBERTA_PATH}/vocab.json", 

    merges_file=f"{ROBERTA_PATH}/merges.txt", 

    lowercase=True,

    add_prefix_space=True

)
from transformers import RobertaTokenizer, RobertaModel

import torch



tokenizer = RobertaTokenizer.from_pretrained('../input/roberta-base')

model = RobertaModel.from_pretrained('../input/roberta-base')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1

outputs = model(input_ids)



print(f"Input_ids = {input_ids}")

print(f"Outputs = {outputs}")
from transformers import RobertaTokenizer, RobertaForMaskedLM

import torch



tokenizer = RobertaTokenizer.from_pretrained('../input/roberta-base')

model = RobertaForMaskedLM.from_pretrained('../input/roberta-base')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1

outputs = model(input_ids, masked_lm_labels=input_ids)

loss, prediction_scores = outputs[:2]



print(f"Input_ids = {input_ids}")

print(f"Outputs = {outputs}")

print(f"Loss = {loss}")

print(f"Prediction Scores = {prediction_scores}")
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import torch



tokenizer = RobertaTokenizer.from_pretrained('../input/roberta-base')

model = RobertaForSequenceClassification.from_pretrained('../input/roberta-base')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)

labels = torch.tensor([1]).unsqueeze(0)

outputs = model(input_ids, labels=labels)

loss, logits = outputs[:2]



print(f"Input_ids = {input_ids}")

print(f"Labels = {labels}")

print(f"Outputs = {outputs}")

print(f"Loss = {loss}")

print(f"Logits = {logits}")
from transformers import RobertaTokenizer, RobertaForTokenClassification

import torch



tokenizer = RobertaTokenizer.from_pretrained('../input/roberta-base')

model = RobertaForTokenClassification.from_pretrained('../input/roberta-base')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)

labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)

outputs = model(input_ids, labels=labels)

loss, scores = outputs[:2]



print(f"Input_ids = {input_ids}")

print(f"Labels = {labels}")

print(f"Outputs = {outputs}")

print(f"Loss = {loss}")

print(f"Scores = {scores}")