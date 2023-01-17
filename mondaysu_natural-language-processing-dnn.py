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
from tensorflow.keras.datasets import imdb

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Flatten,Embedding

from tensorflow.keras.preprocessing import sequence

import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = imdb.load_data()

print(x_train[0])

print(len(x_train[0]))

print(x_train[1])

print(len(x_train[1]))
print(x_train.shape)

print(y_train.shape)

print(y_train[:10])
lens = list(map(len, x_train))

print(np.mean(lens))
plt.hist(lens, bins=range(min(lens), max(lens)+50, 50))

plt.show()
max_length = max(max(list(map(len, x_train))), max(list(map(len, x_test))))

max_length
max_word = 500
dict_size = np.max([np.max(x_train[i]) for i in range(x_train.shape[0])]) + 1

dict_size
x_train = sequence.pad_sequences(x_train, maxlen=max_word)

x_test = sequence.pad_sequences(x_test, maxlen=max_word)
model = Sequential()

model.add(Embedding(dict_size, 128, input_length=max_word))

model.add(Flatten())

model.add(Dense(200, activation='relu'))

model.add(Dense(300, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, 

          validation_data=(x_test, y_test), 

          epochs=10, batch_size=100,

          verbose=0)
score = model.evaluate(x_test, y_test)

# [loss, accuracy]

print(score)