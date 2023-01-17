# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split



from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense,Dropout,Embedding,LSTM,Conv2D,Flatten,MaxPooling2D

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical
# reading Dataset and droping features



data = pd.read_csv("train_file.csv")

data = data.drop(columns=['SentimentTitle','SentimentHeadline'],axis=1)

# data = data.drop(columns=['IDLink', 'Title', 'Source', 'Topic', 'PublishDate','Facebook', 'GooglePlus', 'LinkedIn'],axis=1)
data.head()
x_train = data

y_train = data(columns=[['SentimentTitle','SentimentHeadline']])
# Preprocessing the dataset

tokenizer = Tokenizer(num_words=15000)

tokenizer.fit_on_texts(list(x_train))



X_train = tokenizer.texts_to_sequences(x_train)

X_train = pad_sequences(X_train, maxlen=150)

X_train
y_train=np.array(y_train).reshape((55932,1))
train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size=0.2)
# Creating a LSTM model

model=Sequential()

model.add(Embedding(15000,512,mask_zero=True))

model.add(LSTM(512,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))

model.add(LSTM(256,dropout=0.1, recurrent_dropout=0.1,return_sequences=False))

model.add(Dense(1,activation='softmax'))

model.compile(loss='mean_squared_error',optimizer='Adam',metrics=['mean_squared_error'])

# model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.summary()
# Fitting the dataset into the model

model.fit(train_x[:2000], train_y[:2000], validation_data=(val_x, val_y), epochs=4, batch_size=128, verbose=1)
test = pd.read_csv("test_file.csv")

# test = data.drop(columns=['IDLink', 'Title', 'Source', 'Topic', 'PublishDate','Facebook', 'GooglePlus', 'LinkedIn'],axis=1)
test = tokenizer.texts_to_sequences(test['Headline'])

x_test = pad_sequences(test, maxlen=150)
y = model.predict(x_test)

y = np.argmax(y, axis=1)
# Download predictive value

sub = pd.DataFrame()

sub['SentimentHeadline'] = y

sub.to_csv('output.csv', index=False)