import numpy as np

import pandas as pd

import re

from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences





def loadNclean():

    data = pd.read_csv('../input/Tweets.csv')

    data['text'] = data['text'].apply(lambda x: x.lower())

    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

    return data



max_words = 500

data = loadNclean()

tok = Tokenizer(nb_words=max_words, split=' ')

tok.fit_on_texts(data['text'].values)

X = tok.texts_to_sequences(data['text'].values)

X = pad_sequences(X)
from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM



embed_dim = 32

lstm_out = 10

def buildModel(): 

    model = Sequential()

    model.add(Embedding(max_words,embed_dim,input_length=X.shape[1]))

    model.add(LSTM(lstm_out))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['fbeta_score'])

    return model

model = buildModel()

print(model.summary())
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

Y = pd.get_dummies(data['airline_sentiment']).values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print(X_train.shape,Y_train.shape)
model.fit(X_train,Y_train, nb_epoch=4, batch_size=32, verbose=2)
scores = model.evaluate(X_test, Y_test, verbose=2)

print("Fbeta-score: %.2f" % (scores[1]))