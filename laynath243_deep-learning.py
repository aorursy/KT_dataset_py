from  __future__ import print_function, division

from builtins import range
import os

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, GlobalMaxPool1D

from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model

from sklearn.metrics import roc_auc_score
MAX_SEQUENCE_LENGTH = 100

MAX_VOCAB_SIZE = 20000

EMBEDDING_DIM = 100

VALIDATION_SLIT = 0.2

BATCH_SIZE = 128

EPOCHS = 10
word2vec = {}

with open("../input/glove.6B.100d.txt") as f:

    for line in f:

        values = line.split()

        word = values[0]

        vec = np.asarray(values[1:], dtype='float32')

        word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))
train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")

sentences = train['comment_text'].fillna('DUMMY_VALUE').values

possible_label = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

targets = train[possible_label].values



print("max sequence lenght", max(len(s) for s in sentences))

print("min sequence lenght", min(len(s) for s in sentences))

s = sorted(len(s) for s in sentences)

print("median sequence length", s[len(s) // 2])
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
word2idx = tokenizer.word_index

print('Found %s unique tokens.' % len(word2idx))
data  = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print("Shape of data tensor: ", data.shape)
print('Pre-trained embedings')

num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word2idx.items():

    if i < MAX_VOCAB_SIZE:

        embedding_vector = word2vec.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(

    num_words,

    EMBEDDING_DIM,

    weights = [embedding_matrix],

    input_length = MAX_SEQUENCE_LENGTH,

    trainable = False

)
print('start building ...')

input_ = Input(shape=(MAX_SEQUENCE_LENGTH, ))

x = embedding_layer(input_)

x = Conv1D(128, 3, activation='relu')(x)

x = MaxPooling1D(3)(x)

x = Conv1D(128, 3, activation='relu')(x)

x = MaxPooling1D(3)(x)

x = Conv1D(128, 3, activation='relu')(x)

x = GlobalMaxPool1D()(x)

x = Dense(128, activation='relu')(x)

output = Dense(len(possible_label), activation= 'sigmoid')(x)
model = Model(input_, output)

model.compile(

    loss="binary_crossentropy",

    optimizer='rmsprop',

    metrics=['accuracy']

)
print('Training model...')

r = model.fit(

    data,

    targets,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    validation_split=VALIDATION_SLIT

)