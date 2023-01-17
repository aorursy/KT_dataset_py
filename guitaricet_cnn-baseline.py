import sys

import os

import csv

import time

csv.field_size_limit(sys.maxsize)  # needed for torchtext



import pandas as pd

import numpy as np



import torch

import torch.nn as nn

import torch.nn.functional as F



import torchtext

from tqdm import tqdm



import sklearn.metrics as skm
assert torch.cuda.is_available(), 'We strongly reccomend using GPU for this kernel'
MAX_TITLE_LEN = 20

MAX_BODY_LEN = 1000



index2label = ['news', 'clickbait', 'other']

label2index = {l: i for i, l in enumerate(index2label)}



title_field = torchtext.data.Field(lower=True, include_lengths=False, fix_length=MAX_TITLE_LEN, batch_first=True)

body_field = torchtext.data.Field(lower=True, include_lengths=False, fix_length=MAX_BODY_LEN, batch_first=True)

label_field = torchtext.data.Field(sequential=False, is_target=True, use_vocab=False,

                                   preprocessing=lambda x: label2index[x])



train_dataset = torchtext.data.TabularDataset('../input/train.csv',

                                              format='csv',

                                              fields={'title': ('title', title_field),

                                                      'text': ('body', body_field),

                                                      'label': ('label', label_field)})



val_dataset = torchtext.data.TabularDataset('../input/valid.csv',

                                            format='csv',

                                            fields={'title': ('title', title_field),

                                                    'text': ('body', body_field),

                                                    'label': ('label', label_field)})



test_dataset = torchtext.data.TabularDataset('../input/test.csv',

                                            format='csv',

                                            fields={'title': ('title', title_field),

                                                    'text': ('body', body_field)})



body_field.build_vocab(train_dataset, min_freq=2)

label_field.build_vocab(train_dataset)

vocab = body_field.vocab

title_field.vocab = vocab



print('Vocab size: ', len(vocab))

print(train_dataset[0].title)

print(train_dataset[0].body[:15])

print(train_dataset[0].label)
train_loader, val_loader = torchtext.data.Iterator.splits((train_dataset, val_dataset),

                                                           batch_sizes=(64, 64),

                                                           sort=False,

                                                           device='cuda')



batch = next(iter(train_loader))

print(batch)
class CNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, n_classes,

                 kernel_sizes_cnn, filters_cnn: int, dense_size: int,

                 dropout_rate: float = 0.,):

        super().__init__()



        self._n_classes = n_classes

        self._vocab_size = vocab_size

        self._embedding_size = embedding_size

        self._kernel_sizes_cnn = kernel_sizes_cnn

        self._filters_cnn = filters_cnn

        self._dense_size = dense_size

        self._dropout_rate = dropout_rate



        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)



        self.cnns = []

        for i in range(len(kernel_sizes_cnn)):

            in_channels = embedding_size



            cnn = nn.Sequential(

                nn.Conv1d(in_channels, filters_cnn, kernel_sizes_cnn[i]),

                nn.BatchNorm1d(filters_cnn),

                nn.ReLU()

            )

            cnn.apply(self.init_weights)



            self.add_module(f'cnn_{i}', cnn)

            self.cnns.append(cnn)

        

        # concatenated to hidden to classes

        self.projection = nn.Sequential(

            nn.Dropout(dropout_rate),

            nn.Linear(filters_cnn * len(kernel_sizes_cnn), dense_size),

            nn.BatchNorm1d(dense_size),

            nn.ReLU(),

            nn.Dropout(dropout_rate),

            nn.Linear(dense_size, n_classes)

        )



    @staticmethod

    def init_weights(module):

        if type(module) == nn.Linear or type(module) == nn.Conv1d:

            nn.init.kaiming_normal_(module.weight)



    def forward(self, x):

        x0 = self.embedding(x)

        x0 = torch.transpose(x0, 1, 2)



        outputs0 = []

        outputs1 = []



        for i in range(len(self.cnns)):

            cnn = getattr(self, f'cnn_{i}')

            # apply cnn and global max pooling

            pooled, _ = cnn(x0).max(dim=2)

            outputs0.append(pooled)



        x0 = torch.cat(outputs0, dim=1) if len(outputs0) > 1 else outputs0[0]

        return self.projection(x0)
model = CNN(vocab_size=len(vocab), embedding_size=300, n_classes=3, kernel_sizes_cnn=(1, 2, 3, 5),

            filters_cnn=512, dense_size=256, dropout_rate=0.5)

model.cuda()  # move model to GPU

model
epochs = 3

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



loss_f = F.cross_entropy



for epoch in tqdm(range(epochs)):

    for i, batch in enumerate(train_loader):

        (title, body), label = batch



        logits = model(title)

        loss = loss_f(logits, label)



        loss.backward()

        optimizer.step()

predictions = []

labels = []



# change model mode to 'evaluation'

# disable dropout and use learned batch norm statistics

model.eval()



with torch.no_grad():

    for batch in val_loader:

        (title, body), label = batch

        logits = model(title)



        y_pred = torch.max(logits, dim=1)[1]

        # move from GPU to CPU and convert to numpy array

        y_pred_numpy = y_pred.cpu().numpy()



        predictions = np.concatenate([predictions, y_pred_numpy])

        labels = np.concatenate([labels, label.cpu().numpy()])
skm.f1_score(labels, predictions, average='macro')
# Do not shuffle test set! You need id to label mapping

test_loader = torchtext.data.Iterator(test_dataset, batch_size=128, device='cuda', shuffle=False)



predictions = []



model.eval()



with torch.no_grad():

    for batch in test_loader:

        (title, body), label = batch

        logits = model(title)



        y_pred = torch.max(logits, dim=1)[1]

        # move from GPU to CPU and convert to numpy array

        y_pred_numpy = y_pred.cpu().numpy()



        predictions = np.concatenate([predictions, y_pred_numpy])

predictions_str = [index2label[int(p)] for p in predictions]



# test.csv index in a contiguous integers from 0 to len(test_set)

# to this should work fine

submission = pd.DataFrame({'id': list(range(len(predictions_str))), 'label': predictions_str})

submission.to_csv('submission.csv', index=False)

submission.head()