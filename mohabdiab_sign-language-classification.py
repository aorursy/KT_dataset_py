# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
Image("../input/amer_sign2.png")

train = pd.read_csv('../input/sign_mnist_train.csv')

test = pd.read_csv('../input/sign_mnist_test.csv')
labels = train['label'].values

train.drop('label', axis = 1, inplace = True)
images = train.values

images = np.array([np.reshape(i, (28, 28)) for i in images])

images = np.array([i.flatten() for i in images])
from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

labels = label_binrizer.fit_transform(labels)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)

import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
x_train = x_train / 255

x_test = x_test / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.20))

model.add(Dense(24, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=50, batch_size=128)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title("Accuracy")

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend(['train','test'])

plt.show()
test_labels = test['label']

test.drop('label', axis = 1, inplace = True)

test_images = test.values

test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])

test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

y_pred = model.predict(test_images)

from sklearn.metrics import accuracy_score

accuracy_score(test_labels, y_pred.round())
