# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from torchtext import data, vocab

# Any results you write to the current directory are saved as output.

reviews = pd.read_csv("/kaggle/input/Reviews.csv")
reviews = reviews[['Summary', 'Text']]

print(reviews.shape)

reviews.head()
reviews = reviews.dropna()

reviews = reviews.reset_index(drop=True)

print(reviews.shape)

reviews.iloc[89:91]
%%time

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import spacy

import torchtext

from torchtext.data import Field, BucketIterator, TabularDataset

en = spacy.load('en')



def tokenize(sentence):

    return [tok.text for tok in en.tokenizer(sentence)]



TEXT = Field(tokenize=tokenize,lower=True, eos_token='_eos_')

data_fields = [('Summary', TEXT), ('Text', TEXT)]

train, val = train_test_split(reviews, test_size=0.2)

train.to_csv("train.csv", index=False)

val.to_csv("val.csv", index=False)

train,val = data.TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv', fields=data_fields)
pre_trained_vector_type = 'glove.6B.200d' 

TEXT.build_vocab(train, vectors=pre_trained_vector_type)

train_iter = BucketIterator(train, batch_size=20, sort_key=lambda x: len(x.Text), shuffle=True)
import torch.nn.functional as F

import torch.nn as nn

import torch

class encoder(nn.Module):

    def __init__(self,input_size, embz_size, hidden_size,batch_size, output_size,max_tgt_len,pre_trained_vector, padding_idx,  bias=False):

        super().__init__()

        #Param 

        self.embz_size, self.hidden_size = embz_size, hidden_size//2

        self.input_size, self.max_tgt_len, self.pre_trained_vector = input_size, max_tgt_len, pre_trained_vector

        self.padding_idx = padding_idx

        #Creates DropOut Layer

        self.encoder_dropout = nn.Dropout(0.1)

        #Creates Embd Layer

        self.encoder_embedding_layer = nn.Embedding(self.input_size, self.embz_size, padding_idx=self.padding_idx)

        #If pre trained Vectors exist, copy them to the layer

        if self.pre_trained_vector: self.encoder_embedding_layer.weight.data.copy_(self.pre_trained_vector.weight.data)

        self.encoder_rnn = getattr(nn, 'LSTM')(

                           input_size=self.embz_size,

                           hidden_size=self.hidden_size,

                           num_layers=1,

                           dropout=0, 

                           bidirectional=True)

        self.encoder_vector_layer = nn.Linear(self.hidden_size*2,self.embz_size, bias=bias)

    

    def init_hidden(self, batch_size):

        return ((torch.zeros(2, batch_size, self.hidden_size)), (torch.zeros(2, batch_size, self.hidden_size)))  



    def _cat_directions(self, hidden):

        def _cat(h):

            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)            

        if isinstance(hidden, tuple):

            # LSTM hidden contains a tuple (hidden state, cell state)

            hidden = tuple([_cat(h) for h in hidden])

        return hidden    

    

    

    def forward(self, seq, y=None):

        batch_size = seq[0].size(0)

        encoder_hidden = self.init_hidden(batch_size)

        encoder_input = self.encoder_dropout(self.encoder_embedding_layer(seq))

        encoder_output, encoder_hidden = self.encoder_rnn(encoder_input, encoder_hidden) 

        encoder_hidden = self._cat_directions(encoder_hidden)

        return encoder_hidden

        
input_size = len(TEXT.vocab)

hidden_size = 400

output_size =  len(TEXT.vocab)

max_tgt_len = 200

batch_size = 20

pre_trained_vector = None

enc = encoder(input_size, 200, hidden_size,batch_size, output_size,max_tgt_len,pre_trained_vector, 1)
a = next(iter(train_iter))

enc(a.Text)