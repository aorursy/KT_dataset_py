# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import random

from PIL import Image

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
df_train.head()
df_train.describe()
df_train.shape
df_train.shape
X = df_train.drop(["label"], axis=1).values

y = df_train["label"].values
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print("X_train.shape: {}".format(X_train.shape))

print("y_train.shape: {}".format(y_train.shape))
print("X_test.shape: {}".format(X_test.shape))

print("y_test.shape: {}".format(y_test.shape))
from keras.utils import to_categorical



X = X.reshape((X.shape[0], 28, 28, 1))

X = X.astype('float32') / 255



y = to_categorical(y)



X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

X_train = X_train.astype('float32') / 255



X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

X_test = X_test.astype('float32') / 255



y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
print("X_train.shape: {}".format(X_train.shape))

print("y_train.shape: {}".format(y_train.shape))



print("X_test.shape: {}".format(X_test.shape))

print("y_test.shape: {}".format(y_test.shape))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

from tensorflow.keras import regularizers

import tensorflow as tf
tf.keras.backend.clear_session()



model = Sequential()



model.add(Conv2D(96, (2,2), strides=1, activation='relu', input_shape=(28, 28, 1)))

model.add(MaxPooling2D((2, 2), strides=1))

model.add(BatchNormalization(axis=1))



model.add(Conv2D(96, (2,2), strides=1, activation='relu'))

model.add(MaxPooling2D((3,3), strides=2))

model.add(BatchNormalization(axis=1))



model.add(Conv2D(96, (2,2), strides=1, activation='relu'))

model.add(Conv2D(96, (2,2), strides=1, activation='relu'))

model.add(Conv2D(96, (2,2), strides=1, activation='relu'))

model.add(BatchNormalization(axis=1))



model.add(MaxPooling2D((3,3), strides=2))



model.add(Flatten())



model.add(Dense(32, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.1))

model.add(BatchNormalization())

model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.001)))

model.add(BatchNormalization())

model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.001)))

model.add(BatchNormalization())

model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.001)))

model.add(BatchNormalization())

model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001)))

model.add(BatchNormalization())

model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.001)))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))



model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=128)
loss, accuracy = model.evaluate(X_test, y_test)
model.fit(X, y, batch_size=128, epochs=128)
## Submission:
df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
df_test.head()
X = df_test.values
X.shape
X = X.reshape((28000, 28, 28, 1))

X = X.astype('float32') / 255
X.shape
y = model.predict(X)
y.shape
result = np.argmax(y, axis=1)
result.shape
predictions = pd.DataFrame(result)
predictions.rename(columns={0: "Label"}, inplace=True)
predictions.index
predictions['ImageId'] = predictions.index
cols = ['ImageId', "Label"]



df = predictions[cols]
df["ImageId"] = df['ImageId'] + 1
df.head()
df.tail()
df.to_csv("submission.csv", index=False)