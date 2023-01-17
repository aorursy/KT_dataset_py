# importing the libraries

import pandas as pd

import tensorflow as tf

import numpy as np
# importing the Deep Learning Libraries

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout
# loading the training data

training_data = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')
training_data.head()
# dropping the qid

training_data = training_data.drop(['qid'], axis = 1)
# creating a feature length that contains the total length of the question

training_data['length'] = training_data['question_text'].apply(lambda s: len(s))

# I used a basic way of utilizing a lambda function.
# now checking the mean length of the text for tokenizing the data.

min(training_data['length']), max(training_data['length']), round(sum(training_data['length'])/len(training_data['length']))
training_data[training_data['length'] <= 9]
training_data = training_data.drop(training_data[training_data['length'] <= 9].index, axis = 0)

min(training_data['length']), max(training_data['length']), round(sum(training_data['length'])/len(training_data['length']))
training_data.isnull().sum()
# Tokenizing the text - Converting each word, even letters into numbers. 

max_length = round(sum(training_data['length'])/len(training_data['length']))

tokenizer = Tokenizer(num_words = max_length, 

                      filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',

                     lower = True,

                     split = ' ')
tokenizer.fit_on_texts(training_data['question_text'])
# Actual Conversion takes place here.

X = tokenizer.texts_to_sequences(training_data['question_text'])
print(len(X), len(X[0]), len(X[1]), len(X[2]))
X = pad_sequences(sequences = X, padding = 'pre', maxlen = max_length)

print(len(X), len(X[0]), len(X[1]), len(X[2]))
y = training_data['target'].values

y.shape
# LSTM Neural Network

lstm = Sequential()

lstm.add(Embedding(input_dim = max_length, output_dim = 120))

lstm.add(LSTM(units = 120, recurrent_dropout = 0.2))

lstm.add(Dropout(rate = 0.2))

lstm.add(Dense(units = 120, activation = 'relu'))

lstm.add(Dropout(rate = 0.1))

lstm.add(Dense(units = 2, activation = 'softmax'))



lstm.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
lstm_fitted = lstm.fit(X, y, epochs = 1)
# importing the testing data

testing_data = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')
testing_data.head()
# converting the data into tokens

X_test = tokenizer.texts_to_sequences(testing_data['question_text'])
print(len(X_test), len(X_test[0]), len(X_test[1]), len(X_test[2]))
# paddding the sequences

X_test = pad_sequences(X_test, maxlen = max_length, padding = 'pre')

print(len(X_test), len(X_test[0]), len(X_test[1]), len(X_test[2]))
# predicting the test set

lstm_prediction = lstm.predict_classes(X_test)
# creating a dataframe for submitting

submission = pd.DataFrame(({'qid':testing_data['qid'], 'prediction':lstm_prediction}))
submission.head()
submission.to_csv('submission.csv', index = False)