import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import random

import time

import warnings

warnings.filterwarnings('ignore')

from collections import defaultdict

import itertools



import torch

from torch import nn, optim

from torch.nn import functional as F



cuda = torch.cuda.is_available()

if cuda:

    print("CUDA ON")

else:

    print('NO CUDA')
text = None

for i in range(1,6):

    with open("/kaggle/input/game-of-thrones-book-files/got{}.txt".format(i), 'r') as f:

        if text is None:

            text = f.read()

        else:

            text += f.read()

text = text.replace('\n', '')

len(text)
chars = tuple(set(text)) # drop duplicates

int2char = dict(enumerate(chars))

char2int = {ch: i for i, ch in int2char.items()}

encoded = np.array([char2int[ch] for ch in text[2002:]]) # Number Encoding of Characters

encoded[:100]
def one_hot(arr, n_labels):

    res = np.zeros((arr.size, n_labels), dtype=np.float32)

    res[np.arange(res.shape[0]), arr.flatten()]=1

    res = res.reshape((*arr.shape, n_labels))

    return res



def get_batch(arr, batch_size, seq_length):

    n_batches = len(arr) // (batch_size*seq_length)

    arr = arr[:n_batches*batch_size*seq_length] # So that it divides to zero

    arr = arr.reshape((batch_size, -1))

    for i in range(0, arr.shape[1], seq_length):

        x = arr[:, i:i+seq_length]

        y = np.zeros_like(x) # Labels: the next character

        try:

            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, i+seq_length]

        except IndexError:

            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]

        yield x, y
batches = get_batch(encoded, 8, 50)

x, y = next(batches)

x[0], y[0] # Note that y values are shifted
class LSTM(nn.Module):

    def __init__(self, tokens, n_hiddens, n_layers, drop, lr):

        super(LSTM, self).__init__()

        self.char = tokens

        self.n_hiddens = n_hiddens

        self.n_layers = n_layers

        self.drop = drop

        self.lr = lr

        self.int2char = dict(enumerate(self.char))

        self.char2int = {ch: i for i, ch in self.int2char.items()}

        

        self.LSTM = nn.LSTM(len(tokens), n_hiddens, n_layers, dropout=drop, batch_first=True)

        self.dropout = nn.Dropout(drop)

        self.fc = nn.Linear(n_hiddens, len(self.char))

        

    def forward(self, x, hidden):

        output, hidden = self.LSTM(x, hidden)

        output = self.dropout(output)

        output = output.contiguous().view(-1, self.n_hiddens)

        output = self.fc(output)

        

        return output, hidden

    

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data

        if cuda:

            hidden = (weight.new(self.n_layers, batch_size, self.n_hiddens).zero_().cuda(),

                     weight.new(self.n_layers, batch_size, self.n_hiddens).zero_().cuda())

        else:

            hidden = (weight.new(self.n_layers, batch_size, self.n_hiddens).zero_(),

                     weight.new(self.n_layers, batch_size, self.n_hiddens).zero_())

        return hidden
# ---------------

# Hyperparams

# ---------------



n_hiddens = 512

n_layers = 2

batch_size = 128

seq_length = 100

n_epochs = 30

drop = 0.5

lr = 0.001

clip = 5



model = LSTM(chars, n_hiddens, n_layers, drop, lr)



optimizer = optim.Adam(model.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()



validation = 0.3

val_idx = int(len(encoded)*(1-validation))

train, valid = encoded[:val_idx], encoded[val_idx:]



val_loss_def = np.Inf

tra_losses=[]

val_losses=[]



if cuda:

    model.cuda()

    

for epoch in range(n_epochs):

    

    tra_loss, val_loss = 0,0

    

    h = model.init_hidden(batch_size)

    

    for x, y in get_batch(train, batch_size, seq_length):

        x = one_hot(x, len(chars))

        x, y = torch.from_numpy(x), torch.from_numpy(y)

        if cuda:

            x, y = x.cuda(), y.cuda()

        h = tuple([_.data for _ in h])

        model.zero_grad()

        output, h = model(x, h)

        loss = criterion(output, y.view(-1).long())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        tra_loss += loss.item()

        

    with torch.no_grad():

        h = model.init_hidden(batch_size)

        for x,y in get_batch(valid, batch_size, seq_length):

            x = one_hot(x, len(chars))

            x, y = torch.from_numpy(x), torch.from_numpy(y)

            if cuda:

                x, y = x.cuda(), y.cuda()

            h = tuple([_.data for _ in h])

            model.zero_grad()

            output, h = model(x, h)

            loss = criterion(output, y.view(-1).long())

            val_loss += loss.item()

            

    tra_losses.append(tra_loss)

    val_losses.append(val_loss)

            

    print("Epoch: {}/{}".format(epoch+1, n_epochs))

    print("Training Loss: {:.3f}".format(tra_loss))

    print("Validation Loss: {:.3f}".format(val_loss))

    if val_loss < val_loss_def:

        torch.save(model.state_dict(), 'best_model.pt')

        print("Validation Loss dropped from {:.3f} ---> {:.3f}. Model Saved.".format(val_loss_def, val_loss))

        val_loss_def = val_loss
model.load_state_dict(torch.load('best_model.pt'))



def predict(model, chars, h=None, topk=None):

    

    x = np.array([[model.char2int[chars]]])

    x = one_hot(x, len(model.char))

    x = torch.from_numpy(x)

    if cuda:

        x = x.cuda()

    h = tuple([_.data for _ in h])

    x, h = model(x, h)

    p = F.softmax(x, dim=1).data

    if cuda:

        p = p.cpu()

    if topk is None:

        top_ch = np.arange(len(model.char))

    else:

        p, top_ch = p.topk(topk)

        top_ch = top_ch.numpy().squeeze()

    p = p.numpy().squeeze()

    char = np.random.choice(top_ch, p=p/p.sum())

    return model.int2char[char], h



def sample(model, size, prime, topk):

    

    if cuda:

        model.cuda()

        

    with torch.no_grad():

        chars = [c for c in prime]

        h = model.init_hidden(1)

        

        for c in prime:

            r, h = predict(model, c, h, topk)

        chars.append(r)

        

        for i in range(size):

            r, h = predict(model, chars[-1], h, topk)

            chars.append(r)



    return ''.join(chars)
sample(model, size=1500, prime='The', topk=3)