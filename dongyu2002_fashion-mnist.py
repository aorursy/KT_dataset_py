# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df = pd.DataFrame(pd.read_csv('../input/fashion-mnist_train.csv'))


print (df.head())
TRAINING_SIZE = 40000
one_hot = df.loc[:TRAINING_SIZE - 1, 'label']
one_hot = to_categorical(one_hot, num_classes=10)
X = df.iloc[:TRAINING_SIZE, 1:785]
print (X)
model = Sequential()
model.add(Dense(700, input_dim=784, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(700, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00002), metrics=['accuracy'])
train_op = model.fit(X, one_hot, epochs=100)



evalX = df.iloc[TRAINING_SIZE:, 1:785]
one_hot_eval = df.loc[TRAINING_SIZE:, 'label']
one_hot_eval = to_categorical(one_hot_eval, num_classes=10)

loss_score = model.evaluate(evalX, one_hot_eval)
print ('loss: ', loss_score[0], ' accuracy: ', loss_score[1])
import matplotlib.pyplot as plt

plt.plot(train_op.history['loss'], label='loss')
plt.plot(train_op.history['acc'], label='accuracy')
plt.legend()
plt.show()
test_file = pd.read_csv('../input/fashion-mnist_test.csv')
df_test = pd.DataFrame(test_file)

one_hot_test = df.loc[:, 'label']
one_hot_test = to_categorical(one_hot_test, num_classes=10)

x_Test = df.iloc[:, 1:785]
test_score = model.evaluate(x_Test, one_hot_test)
print ('loss: ', test_score[0], ' accuracy: ', test_score[1])