!pip install pytorch_pretrained_bert
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering, BertForPreTraining ,BertAdam
input_text = "[CLS] I go to school by bus [SEP] "
target_text = "我搭公車上學"
# Load pre-trained model tokenizer (vocabulary)
modelpath = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(modelpath)
model = BertForMaskedLM.from_pretrained(modelpath)
model.to('cuda')

tokenized_text = tokenizer.tokenize(input_text)
for i in target_text:
  tokenized_text.append('[MASK]')
# tokenized_text.append('[SEP]')
for _ in range(128-len(tokenized_text)):
  tokenized_text.append('[MASK]')
# tokenized_text.append('[MASK]')
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')

loss_ids = []
loss_ids = [-1] * (len(tokenizer.tokenize(input_text)))
# loss_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_text)))
for i in target_text:
  loss_ids.append(tokenizer.convert_tokens_to_ids(i)[0])
loss_ids.append(tokenizer.convert_tokens_to_ids(['[SEP]'])[0])
for _ in range(128-len(loss_ids)):
  loss_ids.append(-1)
loss_tensors = torch.tensor([loss_ids]).to('cuda')
print(tokens_tensor,loss_tensors)
print(tokenizer.convert_ids_to_tokens(indexed_tokens))
# param_optimizer = list(model.named_parameters())

# # hack to remove pooler, which is not used
# # thus it produce None grad that break apex
# param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.05},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.01}
#         ]
# optimizer = BertAdam(optimizer_grouped_parameters,
#                              lr=5e-5)

# optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)
# optimizer = torch.optim.SGD(model.parameters(), lr = 5e-5, momentum=0.9)
optimizer = torch.optim.Adamax(model.parameters(), lr = 5e-5)
model.train()
for i in range(0,300):
  loss = model(tokens_tensor,masked_lm_labels=loss_tensors)
  eveloss = loss.mean().item()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  print("step "+ str(i) + " : " + str(eveloss))
model.eval()
with torch.no_grad():
  predictions = model(tokens_tensor)
  start = len(tokenizer.tokenize(input_text))
  while start < len(predictions[0]):
    predicted_index = torch.argmax(predictions[0,start]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
    if '[SEP]' in predicted_token:
        break
    print(predicted_token)
    start+=1
