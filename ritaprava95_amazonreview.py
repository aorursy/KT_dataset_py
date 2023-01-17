# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

import pandas as pd

import re

import string

from string import digits

import keras.backend as K

import tensorflow as tf

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Input, Embedding, SimpleRNN, LSTM, Dropout, Activation

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.optimizers import Adam, SGD, RMSprop

import matplotlib.pyplot as plt

train_data = pd.read_csv("/kaggle/input/amazon/train.csv")





train_title_X = train_data.loc[:,'Review Title']

train_title_X = [x.lower() for x in train_title_X]

#train_title_X = [re.sub("'", '', x) for x in train_title_X]

train_text_X = train_data.loc[:,'Review Text']

train_text_X = [x.lower() for x in train_text_X]

#train_text_X = [re.sub("'", '', x) for x in train_text_X]

train_y = train_data['topic']

train_y = pd.get_dummies(train_y)



t = Tokenizer()

t.fit_on_texts(train_title_X)

title_vocab_size = len(t.word_index) + 1

    

encoded_train_title_X = t.texts_to_sequences(train_title_X)

    

word_title_index = t.word_index



t.fit_on_texts(train_text_X)

text_vocab_size = len(t.word_index) + 1



encoded_train_text_X = t.texts_to_sequences(train_text_X)



word_text_index = t.word_index





max_title_length = max([len(x) for x in encoded_train_title_X])

padded_train_title_X = pad_sequences(encoded_train_title_X, maxlen=max_title_length, padding='post')



max_text_length = max([len(x) for x in encoded_train_text_X])

padded_train_text_X = pad_sequences(encoded_train_text_X, maxlen=max_text_length, padding='post')

    

    

embeddings_index = dict()

words = []

f = open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt', encoding="utf8")

for line in f:

    values = line.split()

    word = values[0]

    words.append(word)

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

    	

f.close()

#print('Loaded %s word vectors.' % len(embeddings_index))

    

embedding_title_matrix = np.zeros((title_vocab_size, 100))

for word, i in word_title_index.items():

   embedding_vector = embeddings_index.get(word)

   if embedding_vector is not None:

       embedding_title_matrix[i] = embedding_vector

       

embedding_text_matrix = np.zeros((text_vocab_size, 100))

for word, i in word_text_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_text_matrix[i] = embedding_vector
sentences_title = Input((max_title_length,), dtype='int32')

embeddings_title = Embedding(title_vocab_size, 100, weights=[embedding_title_matrix], input_length=max_title_length, trainable=False)(sentences_title)

X = LSTM(128, return_sequences=False)(embeddings_title)

X = Dropout(0.2)(X)

X = Dense(21)(X)

X = Activation('softmax')(X)



sentences_text = Input((max_text_length,), dtype='int32')

embeddings_text = Embedding(text_vocab_size, 100, weights=[embedding_text_matrix], input_length=max_text_length, trainable=False)(sentences_text)

X2 = LSTM(128, return_sequences=True)(embeddings_text)

X2 = LSTM(128, return_sequences=False)(X2)

X2 = Dense(21)(X2)

X2 = Activation('softmax')(X2)

    

model = Model(inputs=sentences_title, outputs=X)

model.summary()
epochs=50

lr=.001

decay = lr/epochs

adam = Adam(lr=lr, beta_1=.9, beta_2=.999, decay=decay)

sgd = SGD(lr=.001)

rmsprop = RMSprop(lr=.001)

optimizers = [adam]

for optimizer in optimizers:

    print('Optimizer :'+str(optimizer))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(padded_train_title_X, train_y, validation_split=0.2, epochs=epochs, verbose=1, batch_size=32)

    plt.figure()

    plt.plot(model.history.history['accuracy'])

    plt.plot(model.history.history['val_accuracy'])

    plt.legend(['training', 'validation'])

    plt.xlabel('epochs')

    plt.xlabel('accuracy')

    