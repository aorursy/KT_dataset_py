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
import pandas as pd

import numpy as np

import tensorflow as tf



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)

test = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)



train_y = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None)
train.head()
test.head()
train_y.head()
train.info()
X = train.values

y = train_y.values

X_submission = test.values
print("X.shape: {}".format(X.shape))

print("y.shape: {}".format(y.shape))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



print("X_train.shape: {}".format(X_train.shape))

print("y_train.shape: {}".format(y_train.shape))

print("X_test.shape: {}".format(X_test.shape))

print("y_test.shape: {}".format(y_test.shape))
from sklearn.preprocessing import StandardScaler, MinMaxScaler



std_scaler = StandardScaler()

scaled_std = std_scaler.fit_transform(X)



minmax_scaler = MinMaxScaler()

scaled_minmax = minmax_scaler.fit_transform(X)



X_train_std, X_test_std, y_train, y_test = train_test_split(scaled_std, y, test_size=0.2)

X_train_minmax, X_test_minmax, y_train, y_test = train_test_split(scaled_minmax, y, test_size=0.2)
import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_model.score(X_test, y_test)
xgb_model.fit(X_train_std, y_train)

xgb_model.score(X_test_std, y_test)
xgb_model.fit(X_train_minmax, y_train)

xgb_model.score(X_test_minmax, y_test)
from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier()



knn_model.fit(X_train, y_train)

knn_model.score(X_test, y_test)
knn_model.fit(X_train_std, y_train)

knn_model.score(X_test_std, y_test)
knn_model.fit(X_train_minmax, y_train)

knn_model.score(X_test_minmax, y_test)
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Activation

import keras



keras.backend.clear_session()



nn_model = Sequential()



nn_model.add(Dense(16, input_dim=40, activation='relu'))

nn_model.add(Dense(32, activation='relu'))

nn_model.add(Dense(1, activation='sigmoid'))



nn_model.summary()



nn_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])



nn_model.fit(X_train, y_train, epochs=64, batch_size=10, verbose=0)



nn_model.evaluate(X_test, y_test)
keras.backend.clear_session()



nn_model.fit(X_train_minmax, y_train, epochs=64, batch_size=10, verbose=0)

nn_model.evaluate(X_test_minmax, y_test)
keras.backend.clear_session()



nn_model.fit(X_train_std, y_train, epochs=64, batch_size=10, verbose=0)

nn_model.evaluate(X_test_std, y_test)
xgb_model.fit(X_train_minmax, y_train)

xgb_model.score(X_test_minmax, y_test)
scaled_minmax_sub = minmax_scaler.fit_transform(X_submission)
xgb_pred = xgb_model.predict(scaled_minmax_sub)
xgb_pred = pd.DataFrame(xgb_pred, columns=['Solution'])
xgb_pred = xgb_pred.reset_index()
xgb_pred.rename(columns={'index': 'Id'}, inplace=True)
xgb_pred['Id'] = xgb_pred['Id'] + 1
xgb_pred['Solution'] = xgb_pred['Solution'].astype(int)
xgb_pred.head()
xgb_pred.tail()
xgb_pred.to_csv('Submission.csv', index=False)