# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
top_words = 5000

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=top_words)
X_train
np.unique(y_train)
from tensorflow.keras import Sequential

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.layers import Embedding, Dense, LSTM
max_review_length = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
embedding_vector_length = 32



model = Sequential()

model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=3, batch_size=64)
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

print("Test accuracy: {}%".format(accuracy * 100))
embedding_vector_length = 32



model = Sequential()

model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=3, batch_size=64)
accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test accuracy: {}%".format(accuracy[1] * 100))
from tensorflow.keras.layers import Conv1D, MaxPooling1D



embedding_vector_length = 32



model = Sequential()

model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=3, batch_size=64)
accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test accuracy: {}%".format(accuracy[1] * 100))