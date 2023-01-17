!pip install pytorch-pretrained-bert
!pip install spacy ftfy==4.4.3

!python -m spacy download en
import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook, tqdm

tqdm.pandas('my bar!')



import warnings

warnings.filterwarnings('ignore')
import torch

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM



# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows

import logging

logging.basicConfig(level=logging.INFO)



# Load pre-trained model tokenizer (vocabulary)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')



# Tokenized input

text = "[CLS] Who was Jim Henson ? Jim Henson was a puppeteer [SEP]"

tokenized_text = tokenizer.tokenize(text)



# Convert token to vocabulary indices

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Define sentence

segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



# Convert inputs to PyTorch tensors

tokens_tensor = torch.tensor([indexed_tokens])

segments_tensors = torch.tensor([segments_ids])
tokenized_text
# Load pre-trained model (weights)

model = BertModel.from_pretrained('bert-large-uncased')

model.eval()



# If you have a GPU, put everything on cuda

tokens_tensor = tokens_tensor.to('cuda')

segments_tensors = segments_tensors.to('cuda')

model.to('cuda')
# Predict hidden states features for each layer

with torch.no_grad():

    _, embedding_data = model(tokens_tensor, segments_tensors, output_all_encoded_layers=False)
embedding_data.cpu().numpy()[0][1]
%%time

train_df = pd.read_csv("../input/train.csv")
train_df.head()
MAX_SEQ_LENGTH = 220
def convert_lines(example, max_seq_length, tokenizer):

    max_seq_length -=2

    all_tokens = []

    longer = 0

    for text in tqdm_notebook(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

        # for testing the aactivity of the program

        if longer > 3:

            break

    print(longer)

    return np.array(all_tokens)
tokenized_text = convert_lines(train_df['comment_text'].values, MAX_SEQ_LENGTH, tokenizer)
tokenized_text.shape
def get_BERT_embedding(model, sentence_index):

    # If you have a GPU, put everything on cuda

    tokens_tensor = torch.tensor([sentence_index])

    tokens_tensor = tokens_tensor.to('cuda')

    model.to('cuda')

    # Predict hidden states features for each layer

    with torch.no_grad():

        _, embedding_data = model(tokens_tensor, output_all_encoded_layers=False)

    return embedding_data
def get_BERT_embeddings(model, sentence_indexes):

    embeddings = []

    for i in tqdm_notebook(range(len(sentence_indexes))):        

        embeddings.append(get_BERT_embedding(model, sentence_indexes[i])[0].cpu().numpy())

    return embeddings
embeddings = get_BERT_embeddings(model, tokenized_text)

embeddings
sub = pd.DataFrame({'id' : train_df['id'],

                    'target' : train_df['target'],

                    'embedding' : embeddings})

sub.to_csv('embedding', index=False)
sub