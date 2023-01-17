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

def myfunc(x):
    if x < 30:
        mult = 10
    elif x < 60:
        mult = 20
    else:
        mult = 50
    return x*mult
print(myfunc(10))
print(myfunc(30))
print(myfunc(60))
import numpy as np
x = np.arange(0, 100, .01)

myfuncv = np.vectorize(myfunc)
y = myfuncv(x)
X = x.reshape(-1, 1)
import sklearn.model_selection as sk

X_train, X_test, y_train, y_test = sk.train_test_split(X,y,test_size=0.33, random_state = 42)
print(X_train.shape)
print(X_test.shape)
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train)
plt.scatter(X_test , y_test)
import tensorflow as tf
import numpy as np
print(tf.__version__)
# Import the kera modules
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor. Since the input only has one column
inputs = Input(shape=(1,))

# a layer instance is callable on a tensor, and returns a tensor
# To the first layer we are feeding inputs
x = Dense(32, activation='relu')(inputs)
# To the next layer we are feeding the result of previous call here it is h
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# Predictions are the result of the neural network. Notice that the predictions are also having one column.
predictions = Dense(1)(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mse'])
model.fit(X_train, y_train,  epochs=550, batch_size=64)  # starts training
# x_test = np.arange(0, 100, 0.02)
# X_test = x_test.reshape(-1, 1)
y_test = model.predict(X_test)
plt.scatter(X_test, y_test)
print(y_test)
print(y_test.shape)
