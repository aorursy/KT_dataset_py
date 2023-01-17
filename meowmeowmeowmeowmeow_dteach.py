import numpy as np

import pandas as pd

import os

import sys

import warnings

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential

from keras.layers import Convolution1D

from keras import initializers, regularizers, constraints, optimizers, layers
df = pd.read_csv('../input/dtttttttttttt2//dtttt.csv').sample(frac=1)

df.sample(5)

df = df.sample(frac=1)

tr = df.iloc[:800]

ts = df.iloc[800:]

tokenizer = Tokenizer(num_words=128)
tokenizer.fit_on_texts(tr['text'])
list_tokenized_train = tokenizer.texts_to_sequences(tr['text'])
X_t = pad_sequences(list_tokenized_train, maxlen=128)

y = tr['res']

y = np.array(y)
import sklearn.ensemble

clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=1)

clf.fit(X_t, y)

_y = clf.predict(X_t)

tracc = sklearn.metrics.accuracy_score(_y, y)



list_tokenized_train = tokenizer.texts_to_sequences(ts['text'])

X_ts = pad_sequences(list_tokenized_train, maxlen=128)

ys = ts['res']



_y = clf.predict(X_ts)

tsacc = sklearn.metrics.accuracy_score(_y, ys)

print('Train accuracy:', tracc, 'Test accuracy:', tsacc)