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
Train = pd.read_csv("../input/mnist_train.csv")

Train
Test = pd.read_csv("../input/mnist_test.csv")

Test
Train.shape, Test.shape
img1 = np.array(Train.iloc[5,1:])

img1 = img1.reshape(28,28)

img2 = np.array(Train.iloc[2,1:])

img2 = img2.reshape(28,28)

img3 = np.array(Train.iloc[17,1:])

img3 = img3.reshape(28,28)

img4 = np.array(Train.iloc[4,1:])

img4 = img4.reshape(28,28)
img.shape
import matplotlib.pyplot as plt

plt.imshow(img1, cmap="gray")

plt.show()

plt.imshow(img2, cmap="gray")

plt.show()

plt.imshow(img3, cmap="gray")

plt.show()

plt.imshow(img4, cmap="gray")

plt.show()
X_train = np.array(Train.iloc[:,1:])

Y_train = np.array(Train.iloc[:,0])

X_test = np.array(Test.iloc[:,1:])

Y_test = np.array(Test.iloc[:,0])
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, 
from keras.utils import np_utils

Y_train = np_utils.to_categorical(Y_train, 10)

Y_test = np_utils.to_categorical(Y_test, 10)
Y_train[0]
import keras

from keras.models import Sequential

from keras.layers import Dense, Activation
model = Sequential()

model.add(Dense(256, input_shape=(784,)))

model.add(Activation('relu'))

model.add(Dense(10))

model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, Y_train)
model.evaluate(X_test, Y_test)