import pandas as pd

import transformers

import tokenizers

import torch

import torch.nn as nn
data = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

data.head()
ROBERTA_PATH = '../input/roberta-base'
MAX_LEN = 192

TOKENIZER = tokenizers.ByteLevelBPETokenizer(vocab_file=f"{ROBERTA_PATH}/vocab.json", 

                                             merges_file=f"{ROBERTA_PATH}/merges.txt", 

                                             add_prefix_space=True, 

                                             lowercase=True)
tokens = TOKENIZER.encode(data.text.values[0])

tokens
tokens.tokens
tokens.ids
tokens.type_ids
tokens.offsets
tokens.attention_mask
expt_tokenizer = tokenizers.ByteLevelBPETokenizer(vocab_file=f"{ROBERTA_PATH}/vocab.json", 

                                                merges_file=f"{ROBERTA_PATH}/merges.txt", 

                                                add_prefix_space=False, 

                                                lowercase=True)



temp = expt_tokenizer.encode(data.text.values[0])

temp.tokens, temp.ids, temp.type_ids, temp.offsets, temp.attention_mask
expt_tokenizer.decode(expt_tokenizer.encode("Hello").ids)
TOKENIZER.decode(TOKENIZER.encode("Hello").ids)
conf = transformers.ReformerConfig.from_pretrained(ROBERTA_PATH)

model = transformers.RobertaModel.from_pretrained(ROBERTA_PATH, config=conf)
ids = torch.tensor([[0] + tokens.ids + [2]])

attention_mask = torch.tensor([[1, 1] + tokens.attention_mask])

token_type_ids = torch.tensor([tokens.type_ids + [0, 0]])
output = model(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
len(output)
output[0].shape
output[1].shape
model = transformers.RobertaForQuestionAnswering.from_pretrained('roberta-base')
ques = "What is the name of prime minister of India?"

text = "India is one of the largest country in the world and its current prime minister is Narendra Modi."
tok_ques = TOKENIZER.encode(ques)

tok_text = TOKENIZER.encode(text)
len(tok_ques.ids), len(tok_text.ids)
ids = torch.tensor([[0] + tok_ques.ids + [2, 2] + tok_text.ids + [2]])

attention_mask = torch.tensor([[1] + tok_ques.attention_mask + [1, 1] + tok_text.attention_mask + [1]])

# roberta doesn't make use of token_type_ids so we can have a all zero tensor of correct dimension

token_type_ids = torch.tensor([[0] + tok_ques.type_ids + [0, 0] + tok_text.type_ids + [0]])
start, end = model(ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
start = nn.Softmax()(start)

end = nn.Softmax()(end)
start.shape, end.shape
n = start.shape[1]

max_ij = 0



start_idx = None

end_idx = None



for i in range(14, n-2):

    for j in range(i+1, n-1):

        if start[0][i] + end[0][j] > max_ij:

            max_ij = start[0][i] + end[0][j]

            start_idx = i

            end_idx = j
start_idx, end_idx, max_ij
result = list(ids[0][start_idx: end_idx+1])
TOKENIZER.decode(ids=result)