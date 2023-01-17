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
from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Dense,Flatten

from keras.preprocessing.image import img_to_array, load_img

import matplotlib.pyplot as plt

from glob import glob
df_test = pd.read_csv("../input/mnist_test.csv")

df_train = pd.read_csv("../input/mnist_train.csv")
test = df_test.values

train = df_train.values
y_test = test[:,0].reshape(10000,1)

X_test = test[:,1:785].reshape(10000,28,28,1)

y_train = train[:,0].reshape(60000,1)

X_train = train[:,1:785].reshape(60000,28,28,1)
from keras.utils import to_categorical

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
model.predict(X_test[:4])