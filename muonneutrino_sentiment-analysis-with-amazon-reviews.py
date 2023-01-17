# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.python.keras import models, layers, optimizers

import tensorflow

from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence

from tensorflow.keras.preprocessing.sequence import pad_sequences

import bz2

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import re



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def get_labels_and_texts(file):

    labels = []

    texts = []

    for line in bz2.BZ2File(file):

        x = line.decode("utf-8")

        labels.append(int(x[9]) - 1)

        texts.append(x[10:].strip())

    return np.array(labels), texts

train_labels, train_texts = get_labels_and_texts('../input/train.ft.txt.bz2')

test_labels, test_texts = get_labels_and_texts('../input/test.ft.txt.bz2')
import re

NON_ALPHANUM = re.compile(r'[\W]')

NON_ASCII = re.compile(r'[^a-z0-1\s]')

def normalize_texts(texts):

    normalized_texts = []

    for text in texts:

        lower = text.lower()

        no_punctuation = NON_ALPHANUM.sub(r' ', lower)

        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)

        normalized_texts.append(no_non_ascii)

    return normalized_texts

        

train_texts = normalize_texts(train_texts)

test_texts = normalize_texts(test_texts)
from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(

    train_texts, train_labels, random_state=57643892, test_size=0.2)
MAX_FEATURES = 12000

tokenizer = Tokenizer(num_words=MAX_FEATURES)

tokenizer.fit_on_texts(train_texts)

train_texts = tokenizer.texts_to_sequences(train_texts)

val_texts = tokenizer.texts_to_sequences(val_texts)

test_texts = tokenizer.texts_to_sequences(test_texts)

MAX_LENGTH = max(len(train_ex) for train_ex in train_texts)

train_texts = pad_sequences(train_texts, maxlen=MAX_LENGTH)

val_texts = pad_sequences(val_texts, maxlen=MAX_LENGTH)

test_texts = pad_sequences(test_texts, maxlen=MAX_LENGTH)

def build_model():

    sequences = layers.Input(shape=(MAX_LENGTH,))

    embedded = layers.Embedding(MAX_FEATURES, 64)(sequences)

    x = layers.Conv1D(64, 3, activation='relu')(embedded)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPool1D(3)(x)

    x = layers.Conv1D(64, 5, activation='relu')(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPool1D(5)(x)

    x = layers.Conv1D(64, 5, activation='relu')(x)

    x = layers.GlobalMaxPool1D()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(100, activation='relu')(x)

    predictions = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=sequences, outputs=predictions)

    model.compile(

        optimizer='rmsprop',

        loss='binary_crossentropy',

        metrics=['binary_accuracy']

    )

    return model

    

model = build_model()
model.fit(

    train_texts, 

    train_labels, 

    batch_size=128,

    epochs=2,

    validation_data=(val_texts, val_labels), )
preds = model.predict(test_texts)

print('Accuracy score: {:0.4}'.format(accuracy_score(test_labels, 1 * (preds > 0.5))))

print('F1 score: {:0.4}'.format(f1_score(test_labels, 1 * (preds > 0.5))))

print('ROC AUC score: {:0.4}'.format(roc_auc_score(test_labels, preds)))
def build_rnn_model():

    sequences = layers.Input(shape=(MAX_LENGTH,))

    embedded = layers.Embedding(MAX_FEATURES, 64)(sequences)

    x = layers.CuDNNGRU(128, return_sequences=True)(embedded)

    x = layers.CuDNNGRU(128)(x)

    x = layers.Dense(32, activation='relu')(x)

    x = layers.Dense(100, activation='relu')(x)

    predictions = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=sequences, outputs=predictions)

    model.compile(

        optimizer='rmsprop',

        loss='binary_crossentropy',

        metrics=['binary_accuracy']

    )

    return model

    

rnn_model = build_rnn_model()
rnn_model.fit(

    train_texts, 

    train_labels, 

    batch_size=128,

    epochs=1,

    validation_data=(val_texts, val_labels), )
preds = rnn_model.predict(test_texts)

print('Accuracy score: {:0.4}'.format(accuracy_score(test_labels, 1 * (preds > 0.5))))

print('F1 score: {:0.4}'.format(f1_score(test_labels, 1 * (preds > 0.5))))

print('ROC AUC score: {:0.4}'.format(roc_auc_score(test_labels, preds)))