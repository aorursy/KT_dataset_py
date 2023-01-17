# Install hazm for Tokenize and Normalize



# !pip install hazm

# !pip install https://github.com/sobhe/hazm/archive/master.zip --upgrade
import numpy as np 

import pandas as pd 

import os

import tqdm

from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

import random

from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer as tk

from keras.layers import Dense,Dropout,Embedding,LSTM

from keras.callbacks import EarlyStopping

from keras.losses import categorical_crossentropy

from keras.optimizers import *

from keras.models import Sequential

import spacy



# from __future__ import unicode_literals

# from hazm import *

from bs4 import BeautifulSoup

import re







NLP = spacy.load('en')

#set random seed for the session and also for tensorflow that runs in background for keras

# set_random_seed(123)

random.seed(123)



import warnings

warnings.filterwarnings("ignore", category=UserWarning)
train_csv = pd.read_csv('../input/persian-reviews-with-rate/cmwithrate.csv')

# test = pd.read_csv('../input/persian-reviews-with-rate/test.csv')

trainng = train_csv[train_csv['rate'] <= 2.0]

trainps = train_csv[train_csv['rate'] >= 3.0]

trainps = trainng[:13000]

t = [trainps , trainng]

trainps['rate'] = trainps['rate'].apply(lambda x:1)

trainng['rate'] = trainng['rate'].apply(lambda x:0)

train = pd.concat(t,ignore_index=True)

# train = trainng + trainps

train['rates'] = train['rate'].apply(lambda x:int(x))

train = train.drop(columns='date')

train = train.drop(columns ='rate')

# Show head

train.head(20)
print('neg : ',len(trainng))

print('pos : ',len(trainps))

set(train.rates)
def tokenizer_txt(text):

    text = re.sub(b'\u200c'.decode("utf-8", "strict"), " ", text)   # replace half-spaces with spaces

    text = re.sub('\n', ' ', text)

    text = re.sub('-', '-', text)

    text = re.sub('[ ]+', ' ', text)

    text = re.sub('\.', ' .', text)

    text = re.sub('\،', ' ،', text)

    text = re.sub('\؛ ', ' ؛', text)

    text = re.sub('\؟ ', ' ؟', text)

    text = re.sub('\"',' ',text)

    text = re.sub('\'','',text)

    text = re.sub('\. \. \.', '...', text)

    

    

    return [x.text for x in NLP.tokenizer(text) if text != " "]       
train_tok = train.comment.apply(tokenizer_txt)



X = train_tok.values

rtvalues = list(train.rates.values)

y = to_categorical(train.rates.values)



num_classes = y.shape[1]

# print('Num Class : ', num_classes)

y.shape
y.shape
X_train , X_val , y_train , y_val = train_test_split(X,y,test_size = 0.2 ,stratify = y)



print('Train data Shape : ',X_train.shape)

print('Train target shape :',y_train.shape)

print('valid data shape :',X_val.shape)

print('valid target shape : ',y_val.shape)
len(X_train[0])

# len(X_train)
unique_words = set()

len_max = 0



for sent in tqdm.tqdm_notebook(X_train):

    

    unique_words.update(sent)

    

    if(len_max < len(sent)):

        len_max = len(sent)

        

# Length of the list of unique_words gives the no of unique words



print('We have {} unique word'.format(len(list(unique_words))))

print('The Max Length in words : ',len_max)
max_len = 20

tokenizer = tk(num_words = max_len)

tokenizer.fit_on_texts(list(X_train))



X_train = tokenizer.texts_to_sequences(X_train)

X_val = tokenizer.texts_to_sequences(X_val)

# X_test = tokenizer.texts_to_sequences(test_sentences)



# Padding



X_train = sequence.pad_sequences(X_train , maxlen = max_len)

X_val = sequence.pad_sequences(X_val , maxlen = max_len)



print('shape OF Train : ',X_train.shape)

print('Shape OF valid : ',X_val.shape)

len(X_train[0])
X_train[133]
early_stopping = EarlyStopping(min_delta = 0.001, mode = 'max', monitor='accuracy', patience = 2)

callback = [early_stopping]
model = Sequential()

model.add(Embedding(len(list(unique_words)),300,input_length = max_len))

model.add(LSTM(128 , dropout = 0.2,recurrent_dropout = 0.5,return_sequences = False))

# model.add(LSTM(64,dropout = 0.2, recurrent_dropout = 0.5,return_sequences = False))

model.add(Dense(100 , activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(num_classes , activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',optimizer = Adam(lr = 0.3),metrics = ['accuracy'])

# model.summary()
history=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=50, batch_size=512, verbose=1, callbacks=callback)

        
import matplotlib.pyplot as plt



epoch_count = range(1,len(history.history['loss']) + 1)





plt.plot(epoch_count , history.history['loss'],'r--')

plt.plot(epoch_count, history.history['val_loss'],'b--')



plt.legend(['Training loss','validation loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.show()