# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/fashion-mnist_train.csv")
df_test = pd.read_csv("../input/fashion-mnist_test.csv")
df_train.head()
x_train = df_train.drop('label', axis = 1)
y_train = df_train['label']
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_train
x_train.shape
x_train = x_train.values.reshape(x_train.shape[0], 28, 28, 1)
x_train.shape[1:]
import matplotlib.pyplot as plt
x_train = x_train / 255
plt.imshow(x_train[0].reshape(28,28), cmap='gray')
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=(x_train.shape[1:]), padding='same', activation='relu'))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None))

model.add(Conv2D(128, (3,3), input_shape=(x_train.shape[1:]), padding='same', activation='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))

#output layer - 10 classes
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, validation_split = 0.3, epochs=2, shuffle = True)
df_test.head()
x_test = df_test.drop('label', axis = 1).values
y_test = df_test['label'].values

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_pred = model.predict(x_test)
y_pred
y_pred.shape
y_pred
true_values = []
predicted_values = []

y_test = to_categorical(y_test)

for (true, predicted) in zip(y_test, y_pred):
    true_values.append(true.argmax())
    predicted_values.append(predicted.argmax())    
from sklearn.metrics import accuracy_score
accuracy_score(true_values, predicted_values)


