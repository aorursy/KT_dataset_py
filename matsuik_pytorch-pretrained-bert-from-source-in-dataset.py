# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"))
WORK_DIR = "../working/"

os.listdir(WORK_DIR)
# import module we'll need to import our custom module

import shutil



# copy our file into the working directory

shutil.copytree("../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT/pytorch_pretrained_bert", os.path.join(WORK_DIR, "pytorch_pretrained_bert"))
os.listdir('../input/bert-pretrained-models/cased_l-12_h-768_a-12/cased_L-12_H-768_A-12')
import torch

from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM



# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows

import logging

logging.basicConfig(level=logging.INFO)

BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'

tokenizer = BertTokenizer.from_pretrained(

    BERT_MODEL_PATH, cache_dir=None)



# Tokenized input

text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"

tokenized_text = tokenizer.tokenize(text)



# Mask a token that we will try to predict back with `BertForMaskedLM`

masked_index = 8

tokenized_text[masked_index] = '[MASK]'

print(tokenized_text)
# Convert token to vocabulary indices

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)

segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]



# Convert inputs to PyTorch tensors

tokens_tensor = torch.tensor([indexed_tokens])

segments_tensors = torch.tensor([segments_ids])
os.listdir(BERT_MODEL_PATH)
convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(

    BERT_MODEL_PATH + 'bert_model.ckpt',

BERT_MODEL_PATH + 'bert_config.json',

WORK_DIR + 'pytorch_model.bin')
shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'bert_config.json')
# Load pre-trained model (weights)

model = BertModel.from_pretrained(

    WORK_DIR)

model.eval()



# If you have a GPU, put everything on cuda

tokens_tensor = tokens_tensor.to('cuda')

segments_tensors = segments_tensors.to('cuda')

model.to('cuda')



# Predict hidden states features for each layer

with torch.no_grad():

    encoded_layers, _ = model(tokens_tensor, segments_tensors)

# We have a hidden states for each of the 12 layers in model bert-base-uncased

assert len(encoded_layers) == 12
[e.size() for e in encoded_layers]