# importing necessary libraries 

import pandas as pd

import tensorflow as tf

import os

import re

import numpy as np

from string import punctuation

from zipfile import ZipFile

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# importing neural network libraries

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM, RNN, SpatialDropout1D
train = pd.read_csv('../input/fake-news/train.csv')

test = pd.read_csv('../input/fake-news/test.csv')

train_data = train.copy()

test_data = test.copy()
train_data = train_data.set_index('id', drop = True)
print(train_data.shape)

train_data.head()
print(test_data.shape)

test_data.head()
# checking for missing values

train_data.isnull().sum()
# dropping missing values from text columns alone. 

train_data[['title', 'author']] = train_data[['title', 'author']].fillna(value = 'Missing')

train_data = train_data.dropna()

train_data.isnull().sum()
length = []

[length.append(len(str(text))) for text in train_data['text']]

train_data['length'] = length

train_data.head()
min(train_data['length']), max(train_data['length']), round(sum(train_data['length'])/len(train_data['length']))
len(train_data[train_data['length'] < 50])
train_data['text'][train_data['length'] < 50]
# dropping the outliers

train_data = train_data.drop(train_data['text'][train_data['length'] < 50].index, axis = 0)
min(train_data['length']), max(train_data['length']), round(sum(train_data['length'])/len(train_data['length']))
max_features = 4500
# Tokenizing the text - converting the words, letters into counts or numbers. 

# We dont need to explicitly remove the punctuations. we have an inbuilt option in Tokenizer for this purpose

tokenizer = Tokenizer(num_words = max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')

tokenizer.fit_on_texts(texts = train_data['text'])

X = tokenizer.texts_to_sequences(texts = train_data['text'])
# now applying padding to make them even shaped.

X = pad_sequences(sequences = X, maxlen = max_features, padding = 'pre')
print(X.shape)

y = train_data['label'].values

print(y.shape)
# splitting the data training data for training and validation.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)
# LSTM Neural Network

lstm_model = Sequential(name = 'lstm_nn_model')

lstm_model.add(layer = Embedding(input_dim = max_features, output_dim = 120, name = '1st_layer'))

lstm_model.add(layer = LSTM(units = 120, dropout = 0.2, recurrent_dropout = 0.2, name = '2nd_layer'))

lstm_model.add(layer = Dropout(rate = 0.5, name = '3rd_layer'))

lstm_model.add(layer = Dense(units = 120,  activation = 'relu', name = '4th_layer'))

lstm_model.add(layer = Dropout(rate = 0.5, name = '5th_layer'))

lstm_model.add(layer = Dense(units = len(set(y)),  activation = 'sigmoid', name = 'output_layer'))

# compiling the model

lstm_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
lstm_model_fit = lstm_model.fit(X_train, y_train, epochs = 1)
# GRU neural Network

gru_model = Sequential(name = 'gru_nn_model')

gru_model.add(layer = Embedding(input_dim = max_features, output_dim = 120, name = '1st_layer'))

gru_model.add(layer = GRU(units = 120, dropout = 0.2, 

                          recurrent_dropout = 0.2, recurrent_activation = 'relu', 

                          activation = 'relu', name = '2nd_layer'))

gru_model.add(layer = Dropout(rate = 0.4, name = '3rd_layer'))

gru_model.add(layer = Dense(units = 120, activation = 'relu', name = '4th_layer'))

gru_model.add(layer = Dropout(rate = 0.2, name = '5th_layer'))

gru_model.add(layer = Dense(units = len(set(y)), activation = 'softmax', name = 'output_layer'))

# compiling the model

gru_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
gru_model.summary()
gru_model_fit = gru_model.fit(X_train, y_train, epochs = 1)
print(test.shape)

test_data = test.copy()

print(test_data.shape)
test_data = test_data.set_index('id', drop = True)

test_data.shape
test_data = test_data.fillna(' ')

print(test_data.shape)

test_data.isnull().sum()
tokenizer.fit_on_texts(texts = test_data['text'])

test_text = tokenizer.texts_to_sequences(texts = test_data['text'])
test_text = pad_sequences(sequences = test_text, maxlen = max_features, padding = 'pre')
lstm_prediction = lstm_model.predict_classes(test_text)
submission = pd.DataFrame({'id':test_data.index, 'label':lstm_prediction})

submission.shape
submission.head()
submission.to_csv('submission.csv', index = False)