import numpy as np

import pandas as pd

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, load_model

from keras.layers import Dense, Embedding, LSTM, Bidirectional

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re



data = pd.read_csv("text_emotion.csv")



tokenizer = Tokenizer(num_words=2000, split=' ')



tokenizer.fit_on_texts(data['text'])

X = tokenizer.texts_to_sequences(data['text'])

X = pad_sequences(X)

Y = data['sentiment']



# We can then create our train and test sets:



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
model = Sequential



model.add( Embedding(2000, embed_dim = 128, input_length = X.shape[1], dropout=0.2))

model.add( Bidirectional( LSTM(lstm_out = 196, dropout_U = 0.2, dropout_W = 0.2)))

model.add( Dense(2, activation = 'softmax'))



model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, Y_train, nb_epoch = 7, batch_size = batch_size, verbose = 2)