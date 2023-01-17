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
import matplotlib.pyplot as plt
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
X_train = train.drop(['label'], axis = 1)

y_train = train['label']

X_test = test

X_train = X_train / 255.0

X_test = X_test / 255.0
X_train.shape, y_train.shape, X_test.shape
X_train=X_train.values

X_test=X_test.values

test=test.values







X_train=X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test=X_test.reshape(X_test.shape[0], 28, 28, 1)

test=test.reshape(test.shape[0] , 28 , 28 , 1)
X_train.shape, y_train.shape, X_test.shape
X_train.max(), X_train.min()
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping
input_shape = X_train[0].shape
early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 2, patience = 0)
model1 = Sequential()

model1.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid', activation = 'relu', input_shape = input_shape))

model1.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', activation = 'relu'))

model1.add(MaxPool2D(pool_size=(2,2)))

model1.add(Dropout(0.25))



model1.add(Conv2D(filters = 96, kernel_size = (3,3), padding = 'valid', activation = 'relu'))

model1.add(Conv2D(filters = 128, kernel_size = (5,5), padding = 'valid', activation = 'relu'))

model1.add(MaxPool2D(pool_size=(3,3)))

model1.add(Dropout(0.33))



model1.add(Flatten())

model1.add(Dense(128, activation='relu'))

model1.add(Dropout(0.50))

model1.add(Dense(10, activation='softmax'))



model1.summary()
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(X_train, y_train, epochs = 100, batch_size = 64, verbose = 1, callbacks = [early_stop], shuffle = True)
predictions1 = model1.predict(X_test)
model2 = Sequential()

model2.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid', activation = 'relu', input_shape = input_shape))

model2.add(MaxPool2D(pool_size=(2,2)))

model2.add(Dropout(0.25))



model2.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', activation = 'relu'))

model2.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', activation = 'relu'))

model2.add(MaxPool2D(pool_size=(2,2)))

model2.add(Dropout(0.33))



model2.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'valid', activation = 'relu'))

model2.add(MaxPool2D(pool_size=(2,2)))

model2.add(Dropout(0.4))



model2.add(Flatten())

model2.add(Dense(1024, activation='relu'))

model2.add(Dropout(0.50))

model2.add(Dense(10, activation='softmax'))
model2.summary()
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(X_train, y_train, epochs = 100, batch_size = 128, verbose = 1, callbacks = [early_stop], shuffle = True)
predictions2 = model1.predict(X_test)
model3 = Sequential()

model3.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid', activation = 'relu', input_shape = input_shape))

model3.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'valid', activation = 'relu'))

model3.add(Dropout(0.25))



model3.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', activation = 'relu'))

model3.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'valid', activation = 'relu'))

model3.add(MaxPool2D(pool_size=(2,2)))

model3.add(Dropout(0.33))



model3.add(Conv2D(filters = 96, kernel_size = (3,3), padding = 'valid', activation = 'relu'))

model3.add(Conv2D(filters = 96, kernel_size = (3,3), padding = 'valid', activation = 'relu'))

model3.add(MaxPool2D(pool_size=(2,2)))

model3.add(Dropout(0.4))



model3.add(Flatten())

model3.add(Dense(512, activation='relu'))

model3.add(Dropout(0.50))

model3.add(Dense(1024, activation='relu'))

model3.add(Dropout(0.50))

model3.add(Dense(10, activation='softmax'))
model3.summary()
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model3.fit(X_train, y_train, epochs = 100, batch_size = 128, verbose = 1, callbacks = [early_stop], shuffle = True)
predictions3 = model1.predict(X_test)
predictions = (predictions1 + predictions2 + predictions3)/3

Predictions = []

for i in range (0, len(predictions)):

    Predictions.append(predictions[i].argmax())

id = []

for i in range(1, len(Predictions)+1):

    id.append(i)
output = pd.DataFrame({'ImageId': id, 'Label': Predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
