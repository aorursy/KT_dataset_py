# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.datasets import reuters

from keras.models import Sequential

from keras.layers import Dense, LSTM, Embedding

from keras.preprocessing import sequence

from keras.utils import np_utils



import tensorflow as tf

import matplotlib.pyplot as plt
seed =0

np.random.seed(seed)

tf.random.set_seed(seed)
(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000, test_split=0.2)
category = np.max(Y_train)+1

print(category, '카테고리')

print(len(X_train), '학습용 뉴스 기사')

print(len(X_test), '테스트용 뉴스 기사')

print(X_train[0])

x_train = sequence.pad_sequences(X_train, maxlen=100)

x_test = sequence.pad_sequences(X_test, maxlen=100)

y_train = np_utils.to_categorical(Y_train)

y_test = np_utils.to_categorical(Y_test)
model = Sequential()

model.add(Embedding(1000, 100))

model.add(LSTM(100, activation='tanh'))

model.add(Dense(46, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer = 'adam',

              metrics = ['accuracy'])
history = model.fit(x_train, y_train, batch_size= 100, epochs=20, validation_data=(x_test, y_test))
print("\n Test Accuracy: %.4f"% (model.evaluate(x_test, y_test)[1]))
y_vloss = history.history['val_loss']

y_loss = history.history['loss']
x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker = '.', c="red", label='Testset_loss')

plt.plot(x_len, y_loss, marker = '.', c="blue", label = 'Trainset_loss')



plt.legend(loc='upper right')

plt.grid()

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()