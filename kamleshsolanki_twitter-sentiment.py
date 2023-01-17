import numpy as np

import pandas as pd

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras import Sequential

from keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding, SpatialDropout1D

import re

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/first-gop-debate-twitter-sentiment/Sentiment.csv')
data = data[['text', 'sentiment']]
data['text'] = data['text'].apply(lambda x : x.lower())

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

data['text'] = data['text'].apply(lambda x : x[3:] if x.startswith('rt ') else x[:])
data['sentiment'].value_counts()

data = data[data['sentiment'] != 'Neutral']
tweet_tokenizer = Tokenizer()

tweet_tokenizer.fit_on_texts(data['text'])



sequence = tweet_tokenizer.texts_to_sequences(data['text'])

max_len = max([len(x) for x in sequence])



sequence = pad_sequences(sequence, maxlen=max_len,padding='post')



vocab_size = len(tweet_tokenizer.word_index) + 1
mapping = {'Positive' : 0, 'Negative' : 1}

data['sentiment'] = data['sentiment'].map(mapping)
def build_model(embedding_dim, max_len, vocab_size):

    model = Sequential()

    model.add(Embedding(vocab_size, embedding_dim, input_length = max_len))

    model.add(Bidirectional(LSTM(128, return_sequences = True)))

    model.add(Bidirectional(LSTM(128, return_sequences = False)))

    model.add(Dropout(0.3))

    model.add(Dense(2, activation = 'softmax'))

    

    model.compile(optimizer = 'adadelta', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    

    return model
embedding_dim = 12

model = build_model(embedding_dim, max_len, vocab_size)
X,Y = sequence, data['sentiment']

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.25, shuffle = True, random_state = 7)
hist = model.fit(X_train, Y_train, epochs = 9, validation_data=(X_test, Y_test))
plt.figure(figsize = (15, 7))

for metric in hist.history.keys():

    plt.plot(hist.history[metric], label = metric)

plt.legend()
model.evaluate(X_test, Y_test)