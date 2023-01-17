import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from keras.models import Sequential
from keras.models import load_model, model_from_json
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import time
from datetime import datetime
from matplotlib import pyplot
import pickle
from keras.models import load_model
import keras
df_train_sample = pd.read_csv("../input/train_sample.csv")
df_train = pd.read_csv("../input/train.csv", nrows=10000000) # this might take a while
#df_test = pd.read_csv("../input/test.csv")
df_train.head()
x_train = df_train[['ip', 'app', 'os', 'channel']].values
x_train.shape
y_train = df_train[['is_attributed']].values
y_train.shape
from keras.models import Sequential

model = Sequential()
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=4))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5)
