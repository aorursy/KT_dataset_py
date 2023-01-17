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
import numpy as np

import pandas as pd

import random

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras import backend as K
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 
import time

def prep_data(X_train, y_train, test):

   

    X_train = X_train.astype('float32') / 255

    test = test.astype('float32')/255

    X_train = X_train.values.reshape(-1,28,28,1)

    test = test.values.reshape(-1,28,28,1)

    y_train = keras.utils.np_utils.to_categorical(y_train)

    classes = y_train.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = int(time.time()))

    

    return X_train, y_train, X_test, y_test, classes, test
X_train, y_train, X_test, y_test, out_neurons, test = prep_data(X_train, y_train, test)


model  = Sequential([

        Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape = (28,28,1)),

        Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same'),

        MaxPool2D(pool_size = (2, 2)),

        Dropout(0.25),

        

        Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),

        Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),

        MaxPool2D(pool_size = (2, 2)),

        Dropout(0.25),

        

        Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'),

        Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'),

        MaxPool2D(pool_size = (2, 2)),

        Dropout(0.25),

        

        Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'),

        Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'),

        MaxPool2D(pool_size = (2, 2)),

        Dropout(0.25),

        

        Flatten(),

        

        Dense(512, activation = 'relu'),

        Dropout(0.5),

        Dense(256, activation = 'relu'),

        Dropout(0.5),

        Dense(out_neurons, activation = 'softmax')

    ])

    





from keras import optimizers

model.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss = 'categorical_crossentropy', optimizer = sgd , metrics = ['accuracy'])
model.fit(X_train, y_train,

          batch_size = 512,

          epochs = 180,

          validation_data = (X_test, y_test),

          verbose = 0);
result = model.evaluate(X_test, y_test, verbose = 0)

print('Accuracy: ', result[1])

print('Error: %.2f%%' % (100- result[1]*100))

y_pred = model.predict(test, verbose=0)
solution = np.argmax(y_pred,axis = 1)

solution = pd.Series(solution, name="Label").astype(int)

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),solution],axis = 1)

submission.to_csv("mnist_with_cnn.csv",index=False)