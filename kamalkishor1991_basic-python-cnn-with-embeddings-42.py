import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import sys

from importlib import reload
data_dir = '../input/'

train = pd.read_csv(data_dir + 'train.csv')

test = pd.read_csv(data_dir + 'test.csv')
train.head()
import re

all_words = set([])

train_words = []

test_words = []

train_words_len = []

test_words_len = []

regex = re.compile('[ ]')



def preprocess_sentence(sentence):

    #sentence = sentence.lower()

    punctuations = [',', '.', '"', '\'', '?', ':', ';']

    for p in punctuations:

        sentence = sentence.replace(p, " " + p + " ")

    sentence = re.sub( '\s+', ' ', sentence ).strip()

    return sentence



for sentence in train['text']:

    sentence = preprocess_sentence(sentence)

    words = regex.split(sentence)

    train_words.append(words)

    train_words_len.append(len(words))

    for w in words:

        all_words.add(w)

        

for sentence in test['text']:

    sentence = preprocess_sentence(sentence)

    words = regex.split(sentence)

    test_words.append(words)

    test_words_len.append(len(words))

    for w in words:

        all_words.add(w)

train['word_len'] = train_words_len

test['word_len'] = test_words_len
plt.plot(train['word_len'])

plt.ylabel('text lengths')

plt.show()
train['word_len'].describe()
feature_len = 150
vocab_size = len(all_words) + 1
word2ids = {}

for index, word in enumerate(all_words):

    word2ids[word] = index + 1
x_train = []

x_test = []

y_train = []

for index, row in train.iterrows():

    x = np.zeros(feature_len)

    for ix, word in enumerate(train_words[index]):

        if (ix >= feature_len):

            break

        x[ix] = word2ids[word]

    x_train.append(x)

    y = [0, 0, 0]

    if row['author']=='EAP':

        y[0] = 1

    elif row['author']=='HPL':

        y[1] = 1

    else:

        y[2] = 1

    y_train.append(y)



for index, row in test.iterrows():

    x = np.zeros(feature_len)

    for ix, word in enumerate(test_words[index]):

        if (ix >= feature_len):

            break

        x[ix] = word2ids[word]

    x_test.append(x)

#validation set

msk = np.random.rand(len(x_train)) < 0.95

x_train = np.array(x_train)

y_train = np.array(y_train)

x_valid = x_train[~msk]

y_valid = y_train[~msk]

x_train = x_train[msk]

y_train = y_train[msk]
import keras

from keras.layers import *

from keras.layers.embeddings import Embedding

from keras import regularizers

from keras.models import Model

from keras.optimizers import SGD, RMSprop, Adam

from keras.models import Sequential



from keras.layers import Dense, Activation
import gensim

from gensim.models import *
n_fact = 30
all_sentences = train_words + test_words

word2vec = Word2Vec(all_sentences, size=n_fact, window=7, min_count=2, workers=4, iter=20)
word2vec.wv.similar_by_word('The')
from numpy.random import random, normal



emb = np.zeros((vocab_size, n_fact))

for word,idx in word2ids.items():

    print(word)

    if word in word2vec.wv:

        emb[idx] = word2vec.wv[word]

    else:

        print("word is not present:" + str(word))

        emb[idx] = normal(scale=0.6, size=(n_fact,))
model = Sequential([

    Embedding(vocab_size, n_fact, input_length=feature_len, weights=[emb], trainable=True),

    Dropout(0.5),

    Conv1D(30, 7, border_mode='same', activation='relu'),

    Dropout(0.5),

    Flatten(),

    BatchNormalization(),

    Dense(40, activation='relu'),

    Dropout(0.6),

    Dense(3, activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()
model.fit(np.array(x_train), y_train, validation_data=(np.array(x_valid),y_valid), nb_epoch=3, batch_size=64)
model.fit(np.array(x_train), y_train, validation_data=(np.array(x_valid),y_valid), nb_epoch=3, batch_size=64)
pred = model.predict(np.array(x_test))
submission  = pd.DataFrame()

submission ['id'] = test['id']

submission ['EAP'] = pred[:,0]

submission ['HPL'] = pred[:,1]

submission ['MWS'] = pred[:,2]

submission.to_csv('sub.csv', index=False)
!ls -l