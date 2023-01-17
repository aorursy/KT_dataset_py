# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re

data = pd.read_csv('../input/Sentiment.csv')

data = data[['text', 'sentiment']]

print(data.head)
data = data[data.sentiment != 'Neutral']

data['text'] = data['text'].apply(lambda x: x.lower())

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-Z0-9\s]','',x)))
print(data[data['sentiment'] == 'Positive'].size)

print(data[data['sentiment'] == 'Negative'].size)



for idx, row in data.iterrows():

    row[0] = row[0].replace('rt', ' ')



batch_size = 17500

tokenizer = Tokenizer(nb_words=batch_size, split=' ')

tokenizer.fit_on_texts(data['text'].values)

X = tokenizer.texts_to_sequences(data['text'].values)

X = pad_sequences(X)
embed_dim = 128

lstm_out = 128

model = Sequential()

model.add(Embedding(batch_size, embed_dim, input_length = X.shape[1], dropout=0.2))

model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['fbeta_score'])

print(model.summary())

Y = pd.get_dummies(data['sentiment']).values

Y
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = 42)

print(X_train.shape, Y_train.shape)

print(X_test.shape, Y_test.shape)
model.fit(X_train, Y_train, nb_epoch = 15, batch_size=32, verbose = 2)
validation_size = 1500

X_validate = X_test[-validation_size:]

Y_validate = Y_test[-validation_size:]

X_test = X_test[:-validation_size]

Y_test = Y_test[:-validation_size]

score = model.evaluate(X_test, Y_test, verbose = 2)

print("Score: %2f" %(score[1]))
model.predict(X_validate[1])