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
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation

from tensorflow.keras.layers import Embedding

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Conv1D, MaxPooling1D

from tensorflow.keras.datasets import imdb

import tensorflow as tf

import matplotlib.pyplot as plt
seed = 0

np.random.seed(seed)

tf.random.set_seed(0)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=500)
x_train = sequence.pad_sequences(x_train, maxlen=100)

x_test = sequence.pad_sequences(x_test, maxlen=100)
model = Sequential()

model.add(Embedding(5000, 100))

model.add(Dropout(0.5))

model.add(Conv1D(64, 5, padding='valid', activation ='relu', strides=1))
model.add(LSTM(55))

model.add(Dense(1))

model.add(Activation('sigmoid'))
model.summary()
model.compile(loss = 'binary_crossentropy',

              optimizer = 'adam',

              metrics = ['accuracy'])
history=model.fit(x_train, y_train, batch_size = 100, epochs=5, validation_data=(x_test, y_test))
print("\n Test Accuracy: %.4f"% (model.evaluate(x_test, y_test)[1]))
y_vloss=history.history['val_loss']

y_loss=history.history['loss']
x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c="red", label = 'Testset_loss')

plt.plot(x_len, y_loss, marker = '.', c="blue", label = 'Trainset_loss' )



plt.legend(loc = 'upper right' )

plt.grid()

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()