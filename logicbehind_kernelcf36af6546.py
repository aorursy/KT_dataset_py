# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import keras
from keras.models import Sequential
from keras.layers import Dense
train = pd.read_csv('../input/train.csv')
train.x = train.x / 10000
train.y = train.y /100
train.head()
model = Sequential()
model.add(Dense(units=16, activation='relu', input_dim=1))
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='nadam', loss='mse', metrics=['accuracy'])
model.fit(train.x, train.y, epochs=40, batch_size=128, verbose=2)
y_pred = model.predict(train.x)
plt.plot(train.x, train.y, train.x, y_pred)
test = pd.read_csv('../input/test.csv')
test.x = test.x / 10000
test.y = test.y /100

y_test_pred = model.predict(test.x)
y_test_pred = y_test_pred * 100 

my_submission = pd.DataFrame({'id': test.id, 'y': y_test_pred.flatten()})
my_submission.to_csv('submission.csv', index=False)
