# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
test = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_test.csv')
y_train = train['label']

X_train = train.drop('label',axis=1)

y_test = test['label']

X_test = test.drop('label',axis=1)
X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes=10)

y_test = to_categorical(y_test, num_classes=10)
len(X_train), len(y_train), len(X_test), len(y_test)
from keras import models

from keras import layers

from keras.optimizers import RMSprop
def build_model():

  model = models.Sequential()



  model.add(layers.Conv2D(filters = 32, kernel_size = (5,5), activation ='relu', input_shape = (28,28,1)))

  model.add(layers.MaxPool2D(pool_size=(2,2)))

  model.add(layers.Dropout(0.2))

  model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation ='relu'))

  model.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

  model.add(layers.Dropout(0.2))

  model.add(layers.Flatten())

  model.add(layers.Dense(256, activation = "relu"))

  model.add(layers.Dropout(0.2))

  model.add(layers.Dense(10, activation = "softmax"))



  optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)



  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  return model
model = build_model()

model.fit(X_train, y_train, epochs=30, validation_split=0.2)
cce_score, acc_score = model.evaluate(X_test, y_test)
cce_score, acc_score