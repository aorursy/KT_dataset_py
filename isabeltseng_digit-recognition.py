# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
train.describe()
print('1')

# Any results you write to the current directory are saved as output.
X_train = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')
X_test = pd.read_csv('../input/mnist-in-csv/mnist_test.csv')

y_train = X_train['label']
X_train = X_train.drop(['label'],axis = 1)
y_test = X_test['label']
X_test = X_test.drop(['label'],axis = 1)
X_train.head()
print(X_train.shape)
# print("X_train.shape={}, y_train.shape={}".format(X_train.shapey_ty_train.shape))

# print("X_test.shape={}, y_test.shape={}".format(X_test.shapey_ty_test.shape))
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

X_train = X_train / 255
X_test = X_test / 255                     
print(X_train.shape)                  
print(X_test.shape)
# print(“X_train.shape={}".format(X_train.shape))
# print(“X_test.shape={}".format(X_test.shape))

y_train = np_utils.to_categorical(y_train)
# y_test_categories = y_test
y_test = np_utils.to_categorical(y_test)
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history=model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), validation_split=0.2, epochs=10, batch_size=300, verbose=2)
print('1')
scores = model.evaluate(X_test, y_test)
scores[1]
test = test.values.reshape(-1, 28, 28, 1)
test = test / 255 

results = model.predict(test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("result.csv", index=False)
