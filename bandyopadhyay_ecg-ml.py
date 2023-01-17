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
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



d1 = pd.read_csv("../input/mitbih_train.csv", header=None)

d2 = pd.read_csv("../input/mitbih_test.csv", header=None)

d = pd.concat([d1,d2], axis=0)

x = d.values[:,:-1]

Y = d.values[:, -1].astype(int)

y = to_categorical(Y)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

print(len(x), len(X_train), len(X_test), len(y_train), len(y_test), X_train.shape, y_train.shape)
from keras.models import Sequential

from keras.layers import Dense, Activation



model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(187,)))

model.add(Dense(50, activation='relu'))

model.add(Dense(5, activation='softmax'))



model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=20)



print("Evaluation: ")

mse, acc = model.evaluate(X_test, y_test)

print('mean_squared_error :', mse)

print('accuracy:', acc)
import xgboost as xgb

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.33)

print(len(x), len(Y), len(X_train), len(X_test), len(y_train), len(y_test), X_train.shape, y_train.shape)

params = {'max_depth':7, 'learning_rate':0.1, 'n_estimators':500,'objective':'multi:softprob', 'num_class':5}

dTrain = xgb.DMatrix(X_train, label=y_train)

dTest = xgb.DMatrix(X_test)

clf = xgb.train(params, dTrain, num_boost_round=100)

test_y_predict = clf.predict(dTest)

test_y = np.asarray([np.argmax(row) for row in test_y_predict])

print (accuracy_score(y_test, test_y))