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
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, SpatialDropout1D

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import pickle
import seaborn as sns
import re

tweets = pd.read_csv('../input/train_kaggle.csv', encoding='latin1')
tweets.head(25)

tweets['SentimentText'] = tweets['SentimentText'].map(lambda x: str(x).lower())
tweets['SentimentText'] = tweets['SentimentText'].map(lambda x: re.sub('[^a-z0-9\s]', '', x))
tweets.head()
sns.countplot(x='Sentiment', data = tweets)
max_features = 2000

tokenizer = Tokenizer(num_words = max_features, split = ' ')
tokenizer.fit_on_texts(tweets['SentimentText'].values)

tweet_data = tokenizer.texts_to_sequences(tweets['SentimentText'].values)

from keras.preprocessing.sequence import pad_sequences

tweet_data = pad_sequences(tweet_data)

embed_dim = 128
lstm_out = 128
model= Sequential()
model.add(Embedding(max_features, embed_dim, input_length = tweet_data.shape[1]))
# model.add(Conv1D(64, 5, activation = 'relu'))
# model.add(MaxPooling1D(pool_size= 2))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='sigmoid'))
# model.add(Conv1D(128, 5, activation = 'softmax'))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
# # model.compile(optimizer='adam',
# #               loss='binary_crossentropy',
# #               metrics=['accuracy'])

# model.compile(loss='mean_squared_error', optimizer='sgd',
#               metrics = ['accuracy'])
print (model.summary())
Y = pd.get_dummies(tweets['Sentiment']).values

X_train, X_test, y_train, y_test = train_test_split(tweet_data, Y,
                                                   test_size = 0.2,
                                                   random_state = 42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
partial_X_train = X_train
partial_Y_train = y_train

partial_X_test = X_test
partial_y_test = y_test
batch_size = 1024
model.fit(partial_X_train, partial_Y_train,
         epochs = 3,
         batch_size = batch_size,
         validation_data = (partial_X_test, partial_y_test),
         verbose = 2)
model.save("tweet_sentiment_model.hdf5")
trained_model = load_model("tweet_sentiment_model.hdf5")
test_tweets = pd.read_csv('../input/test_data_kaggle.csv', encoding='latin1')
test_tweets['SentimentText'] = test_tweets['SentimentText'].map(lambda x: str(x).lower())
test_tweets['SentimentText'] = test_tweets['SentimentText'].map(lambda x: re.sub('[^a-z0-9\s]', '', x))
test_tweets.head()

max_features = 2000

tokenizer = Tokenizer(num_words = max_features, split = ' ')
tokenizer.fit_on_texts(test_tweets['SentimentText'].values)

tweet_data = tokenizer.texts_to_sequences(test_tweets['SentimentText'].values)
tweet_data = pad_sequences(tweet_data, maxlen=84)

# Replace the 'Sentiment' column with your predictions
# Convert both columns to string using .astype(str)
# Save the submission dataframe to csv with .to_csv("filename", index=False)
# print(classification_report((Y), predictions))
# print(confusion_matrix((Y), predictions))
predictions = trained_model.predict_classes(tweet_data)
new_df = pd.read_csv('../input/Sample.csv')
# Load in the sample csv
new_df['Sentiment'] = predictions
new_df = new_df.astype(str)
new_df.to_csv("SubmissionNew2.csv", index = False)
!ls
