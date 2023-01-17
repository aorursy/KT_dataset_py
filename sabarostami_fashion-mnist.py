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
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D 

from tensorflow.keras.layers import MaxPooling2D
train_data = np.loadtxt("../input/fashion-mnist_train.csv", delimiter=',',

                       skiprows=1)

x_train = train_data[:,1:]

y_train = train_data[:,0]



test_data = np.loadtxt("../input/fashion-mnist_test.csv", delimiter=',',

                      skiprows=1)

x_test = test_data[:,1:]

y_test = test_data[:,0]



y_train = keras.utils.to_categorical(y_train, 10)

y_test = keras.utils.to_categorical(y_test, 10)



x_train = x_train.reshape(x_train.shape[0], 28,28,1)/225

x_test = x_test.reshape(x_test.shape[0], 28,28,1)/225
model = Sequential()

model.add(Conv2D(32,(3,3), padding="same", activation="relu", 

                 input_shape=(28,28, 1)))

model.add(Conv2D(32,(3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(64,(3,3), padding="same", activation="relu"))

model.add(Conv2D(64,(3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(512, activation="relu"))

model.add(Dropout(0.5)) 

model.add(Dense(10, activation="softmax"))



model.compile(loss="categorical_crossentropy", optimizer="adam", 

              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=32, epochs=10,

          validation_data=(x_test, y_test), shuffle=True)
model.evaluate(x_test, y_test, batch_size=100)