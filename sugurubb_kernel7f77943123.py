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
import re

import numpy as np

import chainer

from chainer import Chain, optimizers, training

from chainer.training import extensions

import chainer.functions as F

import chainer.links as L
import pickle



with open('../input/bert-embedding/BERT_newsheader_1to1000.pickle', 'rb') as handle:

    ds1 = pickle.load(handle)
## eliminate all empty space 



for t in range(0, 1000):

    ds1[t] = [x for x in ds1[t] if x is not None]



#### For experience purpose, I will run with small data set



data_x_vec = []

data_y = []



for t in ds1:

    data_y.append(t[0])

    data_x_vec.append(t[1:-1])

    

### Front_badding for each vector [0]*768



max_sentence_size = 0

for sentence_vec in data_x_vec:

    if max_sentence_size <= len(sentence_vec):

        max_sentence_size = len(sentence_vec)

for sentence_ids in data_x_vec:

    while len(sentence_ids) < max_sentence_size:

        sentence_ids.insert(0, [0]*768) # 先頭に追加



# print(max_sentence_size)
# dataset

data_x_vec = np.array(data_x_vec, dtype="float32")

data_t = np.array(data_y, dtype="int32")

dataset = []

for x, t in zip(data_x_vec, data_t):

    #print(type(x))

    dataset.append((x, t))
# Defining Model

class LSTM_BERT_news_classification(Chain):



    def __init__(self, input_size, hidden_size, out_size):



        super(LSTM_BERT_news_classification, self).__init__(

            eh = L.LSTM(input_size, hidden_size),

            ## For classfier

            hy = L.Linear(hidden_size, out_size)

        )

 

    def __call__(self, x):

        x = F.transpose_sequence(x)

        self.eh.reset_state()

        for BERTVec in x:

            h = self.eh(BERTVec)

                       

        # classification

        y = self.hy(h)

        return y
# parameter

N = len(data_x_vec)

EPOCH_NUM = 5

INPUT_SIZE = 768

HIDDEN_SIZE = 1000

BATCH_SIZE = 8

OUT_SIZE = 2



# Define the model

model = L.Classifier(LSTM_BERT_news_classification(

    input_size=INPUT_SIZE,

    hidden_size=HIDDEN_SIZE,

    out_size=OUT_SIZE

))



# model.compute_accuracy = False
optimizer = optimizers.Adam()

optimizer.setup(model)



# 学習開始

train, test = chainer.datasets.split_dataset_random(dataset, N-100)

train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE)

test_iter = chainer.iterators.SerialIterator(test, BATCH_SIZE, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=-1)

trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")

trainer.extend(extensions.Evaluator(test_iter, model, device=-1))

trainer.extend(extensions.LogReport(trigger=(1, "epoch")))

trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"]))

trainer.run()