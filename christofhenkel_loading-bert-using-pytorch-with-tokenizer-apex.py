# VERSION SUMMARY



# version 6: small bugfix

# version 5: added example for tokenization and prediction

# version 4: added apex install for mixed precision training 
import numpy as np 

import pandas as pd 

import os

import torch
!cd ../input/apex-master/apex-master/apex-master/ && pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
os.system('pip install --no-index --find-links="../input/pytorchpretrainedbert/" pytorch_pretrained_bert')
from pytorch_pretrained_bert import BertTokenizer

from pytorch_pretrained_bert.modeling import BertModel
BERT_FP = '../input/torch-bert-weights/bert-base-uncased/bert-base-uncased/'
bert = BertModel.from_pretrained(BERT_FP).cuda()

bert.eval()
tokenizer = BertTokenizer(vocab_file='../input/torch-bert-weights/bert-base-uncased-vocab.txt')
# lets tokenize some text (I intentionally mispelled 'plastic' to check berts subword information handling)

text = 'hi my name is Dieter and I like wearing my yellow pglastic hat while coding.'

tokens = tokenizer.tokenize(text)

tokens
# added start and end token and convert to ids

tokens = ["[CLS]"] + tokens + ["[SEP]"]

input_ids = tokenizer.convert_tokens_to_ids(tokens)

input_ids
# put input on gpu and make prediction

bert_output = bert(torch.tensor([input_ids]).cuda())

bert_output
import apex

bert.half()