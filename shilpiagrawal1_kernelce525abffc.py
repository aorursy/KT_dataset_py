# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

from keras.layers.embeddings import Embedding







# NLTK

import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer



# Other

import re

import string

import numpy as np

import pandas as pd

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn import metrics





df = pd.read_csv("../input/train.csv")

train_df, val_df = train_test_split(df, test_size=0.1, random_state=2018)

vocabulary_size = 50000



t = Tokenizer(num_words=vocabulary_size)

t.fit_on_texts(train_df['question_text'])

# integer encode the documents

# pad documents to a max length of 4 words

max_length = 100

train_X = t.texts_to_sequences(train_df['question_text'])

val_X = t.texts_to_sequences(val_df['question_text'])



## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=max_length,padding='post')

val_X = pad_sequences(val_X, maxlen=max_length,padding='post')



## Get the target values

train_y = train_df['target'].values

val_y = val_df['target'].values



embeddings_index = {}

f = open(r'../input/embeddings/glove.840B.300d/glove.840B.300d.txt',encoding="utf-8")

for line in f:

    values = line.split(' ')

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



# create a weight matrix for words in training docs

embedding_matrix = np.zeros((vocabulary_size, 300))

for word, index in t.word_index.items():

    if index > vocabulary_size - 1:

        break

    else:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[index] = embedding_vector



def create_conv_model():

    model_conv = Sequential()

    model_conv.add(Embedding(vocabulary_size, 300, input_length=100))

    model_conv.add(Dropout(0.2))

    model_conv.add(Conv1D(64, 5, activation='relu'))

    model_conv.add(MaxPooling1D(pool_size=4))

    model_conv.add(LSTM(300))

    model_conv.add(Dense(1, activation='sigmoid'))

    model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model_conv

#develop a model

model_glove = Sequential()

model_glove.add(Embedding(vocabulary_size, 300, input_length=100, weights=[embedding_matrix], trainable=False))

model_glove.add(Dropout(0.2))

model_glove.add(Conv1D(64, 5, activation='relu'))

model_glove.add(MaxPooling1D(pool_size=4))

model_glove.add(LSTM(300))

model_glove.add(Dense(1, activation='sigmoid'))

model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



labels = df['target']



model_glove.fit(train_X, train_y, batch_size=512,epochs = 2,validation_data=(val_X, val_y))

# Any results you write to the current directory are saved as output.
print(model_glove.summary())

pred_noemb_val_y = model_glove.predict([val_X], batch_size=1024, verbose=1)

from sklearn import metrics

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))))
test_df = pd.read_csv("../input/test.csv")

test_X = t.texts_to_sequences(test_df['question_text'])

test_X = pad_sequences(test_X, maxlen=max_length,padding='post')



pred_test_y = model_glove.predict([test_X], batch_size=1024, verbose=1)

pred_test_y = (pred_test_y>0.35).astype(int)

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)
