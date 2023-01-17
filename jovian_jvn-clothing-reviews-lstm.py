import numpy as np 

import pandas as pd 

import os

import spacy

import string

import re

import numpy as np

from spacy.symbols import ORTH

from collections import Counter



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
reviews = pd.read_csv("reviews.csv")
reviews.head()
reviews.shape
reviews = reviews.loc[~reviews['Review Text'].isna()]
re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)

def sub_br(x): return re_br.sub("\n", x)



my_tok = spacy.load('en')

def spacy_tok(x): return [tok.text for tok in my_tok.tokenizer(sub_br(x))]
counts = Counter()

for index, row in reviews.iterrows():

    #print(row['Review Text'])

    counts.update(spacy_tok(row['Review Text']))
len(counts.keys())
for word in list(counts):

    if counts[word] < 2:

        del counts[word]

vocab2index = {"":0, "UNK":1}

words = ["", "UNK"]

for word in counts:

    vocab2index[word] = len(words)

    words.append(word)

#vocab2index
# note that spacy_tok takes a while run it just once

def encode_sentence(text, vocab2index, N=400, padding_start=True):

    x = spacy_tok(text)

    enc = np.zeros(N, dtype=int)

    enc1 = np.array([vocab2index.get(w, vocab2index["UNK"]) for w in x])

    l = min(N, len(enc1))

    if padding_start:

        enc[:l] = enc1[:l]

    else:

        enc[N-l:] = enc1[:l]

    return enc, l
reviews['encoded'] = reviews['Review Text'].apply(lambda x: np.array(encode_sentence(x,vocab2index )))
X = list(reviews['encoded'])

y = list(reviews['Rating'])

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
Counter(y_valid)
class ReviewsDataset(Dataset):

    def __init__(self, X, Y):

        self.X = X

        self.y = Y

        

    def __len__(self):

        return len(self.y)

    

    def __getitem__(self, idx):

        #return self.X[idx], self.y[idx]

        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx]
train_ds = ReviewsDataset(X_train, y_train)

valid_ds = ReviewsDataset(X_valid, y_valid)
# for x, y in train_ds:

#     print(x)
class LSTMV0Model(torch.nn.Module) :

    def __init__(self, vocab_size, embedding_dim, hidden_dim) :

        super(LSTMV0Model,self).__init__()

        self.hidden_dim = hidden_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.linear = nn.Linear(hidden_dim, 5)

        self.dropout = nn.Dropout(0.5)

        

    def forward(self, x):

        x = self.embeddings(x)

        x = self.dropout(x)

        lstm_out, (ht, ct) = self.lstm(x)

        print(ht.shape)

        print(ht[-1])

        return self.linear(ht[-1])
def train_epocs_v0(model, epochs=10, lr=0.001):

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(parameters, lr=lr)

    for i in range(epochs):

        model.train()

        sum_loss = 0.0

        total = 0

        for x, y in train_dl:

            # s is not used in this model

            x = x.long()#.cuda()

            y = y.long()#.cuda()

            y_pred = model(x)

            print(y_pred)

            optimizer.zero_grad()

            loss = F.cross_entropy(y_pred, y)

            loss.backward()

            optimizer.step()

            sum_loss += loss.item()*y.shape[0]

            total += y.shape[0]

        val_loss, val_acc = val_metrics_v0(model, val_dl)

        if i % 5 == 1:

            print("train loss %.3f val loss %.3f and val accuracy %.3f" % (sum_loss/total, val_loss, val_acc))
def val_metrics_v0(model, valid_dl):

    model.eval()

    correct = 0

    total = 0

    sum_loss = 0.0

    for x, s, y in valid_dl:

        # s is not used here

        x = x.long()#.cuda()

        y = y.long()#.cuda()

        y_hat = model(x)

        loss = F.cross_entropy(y_hat, y)

        pred = torch.max(out, 1)[1]

        correct += (y_pred.float() == y).float().sum()

        total += y.shape[0]

        sum_loss += loss.item()*y.shape[0]

    return sum_loss/total, correct/total



batch_size = 5000

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_dl = DataLoader(valid_ds, batch_size=batch_size)

vocab_size = len(words)

print(vocab_size)

model_v0 = LSTMV0Model(vocab_size, 50, 50)
train_epocs_v0(model_v0, epochs=30, lr=0.01)
import jovian
jovian.commit()