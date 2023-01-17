# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Retrieve training data, X and Y
#train_x dimension: (58784, 93), 93 is the feature number, 58784 is the sample size 
#train_y dimension: (58784, 1)
#This part of code is inspired by: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
seed = 5
df = pd.read_csv('../input/train.csv')
dataset = df.values
X = dataset[:, 1:94].astype(float)
Y = dataset[:, 94]
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
one_hot_Y = to_categorical(encoded_Y)
train_x, test_x, train_y, test_y = train_test_split(X, one_hot_Y, test_size=0.05, random_state=seed)
# Now we have training data all set, start to build model using keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=93))
model.add(Dense(units=48, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=24, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=9, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=100, batch_size=32)
scores = model.evaluate(test_x, test_y)
#print(model.metrics_names)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
classes = model.predict(test_x, batch_size=32)
print(classes)