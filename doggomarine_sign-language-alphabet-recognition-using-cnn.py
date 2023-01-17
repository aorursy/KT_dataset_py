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
train = pd.read_csv("../input/sign_mnist_train.csv")
validation_set = pd.read_csv("../input/sign_mnist_test.csv")
train.head()
import matplotlib.pyplot as plt
import tensorflow as tf
import random
%matplotlib inline
from sklearn.preprocessing import LabelBinarizer
lbtrain = LabelBinarizer()
lbvalid = LabelBinarizer()
#y = np_utils.to_categorical(train['label'].values)
#y_validation = np_utils.to_categorical(validation_set['label'].values)
y = lbtrain.fit_transform(train['label'].values)
y_validation = lbvalid.fit_transform(validation_set['label'].values)

train.drop('label', axis=1, inplace=True)
validation_set.drop('label', axis=1, inplace=True)

x = train.values / 255
x_validation = validation_set.values / 255
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_validation = x_validation.reshape(x_validation.shape[0], 28, 28, 1)
letters = "ABCDEFGHIKLMNOPQRSTUVWXY"
fig = plt.figure(figsize=(20, 5))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))
    ax.set_title("{}".format(letters[np.argmax(y_train[i])]))
plt.show()
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
classes = 24
epochs = 10
batch_size = 150
model = Sequential()

model.add(Conv2D(64, kernel_size=4, activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=4, activation='relu', strides=2))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=4, activation='relu', strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(classes, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
import time
start = time.time()
history = model.fit(x_train, y_train, 
                   validation_data=(x_validation, y_validation),
                   epochs=epochs,
                   batch_size=batch_size)
training_time = time.time() - start
print("Time taken to train model: ", training_time)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy plot')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'])
plt.show()
prediction = model.predict(x_test)
from sklearn.metrics import accuracy_score
print("Test Accuracy: " + str(accuracy_score(y_test, prediction.round())))
