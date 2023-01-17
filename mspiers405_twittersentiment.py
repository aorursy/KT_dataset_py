import keras
import re
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

import pickle
import seaborn as sns
#import data
train_data = pd.read_csv(r'..//input/train_kaggle.csv', encoding="cp1252")
test_data = pd.read_csv(r'..//input/test_data_kaggle.csv', encoding="cp1252")
train_data.head()
#clean data
train_data['SentimentText'] = train_data['SentimentText'].apply(lambda x: x.lower())
train_data['SentimentText'] = train_data['SentimentText'].apply(lambda x: re.sub('[^a-z0-9\s]', '', x))
train_data.head(10)

test_data['SentimentText'] = test_data['SentimentText'].apply(lambda x: x.lower())
test_data['SentimentText'] = test_data['SentimentText'].apply(lambda x: re.sub('[^a-z0-9\s]', '', x))
print(train_data.groupby(['Sentiment']).size())
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(train_data['SentimentText'].values)
X = tokenizer.texts_to_sequences(train_data['SentimentText'].values)
X = pad_sequences(X)
print(X[0])
print('')
tokenizer_test = Tokenizer(num_words=max_features, split=' ')
tokenizer_test.fit_on_texts(test_data['SentimentText'].values)
test_X = tokenizer_test.texts_to_sequences(test_data['SentimentText'].values)
test_X = pad_sequences(test_X, maxlen=84)
print(test_X[0])
Y = pd.get_dummies(train_data['Sentiment'].values)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state=42)
print(X_train.shape)

batch_size = 1000
batch_size = 1000
embed_dim = 32
lstm_out = 48

model = Sequential()

model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(Conv1D(40, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, 
          epochs = 8, 
          batch_size=batch_size, 
          validation_data=(X_test, y_test))
print(test_X[:5])
predicted = model.predict_classes(test_X)

print(predicted)
new_df = pd.DataFrame(data=test_data['Id'],columns=['Id'])
new_df.head()
new_df['Sentiment'] = predicted
new_df.head()
new_df = new_df.astype(str)
new_df.to_csv('submission_kaggle.csv', index=False)