# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# Any results you write to the current directory are saved as output.
#test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")

traincorr = train.corr()['SalePrice']

traincorr
used_columns = ["OverallQual", "GrLivArea", "GarageCars", "SalePrice"]

base = pd.read_csv("../input/train.csv", usecols=used_columns )

base.head()
from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()

base[[

    "OverallQual", 

    "GrLivArea", 

    "GarageCars"

 ]] = scaler_x.fit_transform(base[[

    "OverallQual", 

    "GrLivArea", 

    "GarageCars"

 ]])

base.head()
scaler_y = StandardScaler()

base[['SalePrice']] = scaler_y.fit_transform(base[['SalePrice']])

base.head()
x = base.drop('SalePrice', axis=1)

y = base.SalePrice
x.head()
y.head()
prev_columns = used_columns[0:3]

prev_columns
import tensorflow as tf

columns = [tf.feature_column.numeric_column(key=c) for c in prev_columns]

columns
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train.shape
x_test.shape
function_train = tf.estimator.inputs.pandas_input_fn(x = x_train, y = y_train,

                                                        batch_size=32,

                                                        num_epochs=None,

                                                        shuffle=True)
function_test = tf.estimator.inputs.pandas_input_fn(x = x_test, y = y_test,

                                                        batch_size=32,

                                                        num_epochs=100,

                                                        shuffle=False)
regressor = tf.estimator.LinearRegressor(feature_columns=columns, model_dir='novo')
regressor.train(input_fn=function_train, steps=100)
metrics_train = regressor.evaluate(input_fn=function_train, steps=100)
metrics_test = regressor.evaluate(input_fn=function_test, steps=100)
metrics_train
metrics_test
function_predict = tf.estimator.inputs.pandas_input_fn(x = x_test, shuffle=False)
predict = regressor.predict(input_fn=function_predict)

list(predict)
predict_values = [] 

for p in regressor.predict(input_fn=function_predict):

    predict_values.append(p['predictions'])
predict_values = np.asarray(predict_values).reshape(-1,1)

predict_values
predict_values = scaler_y.inverse_transform(predict_values)

predict_values
y_test2 = scaler_y.inverse_transform(y_test.values.reshape(-1,1))

y_test2
import matplotlib.pyplot as plt

%matplotlib inline

plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.scatter(train['OverallQual'], train['SalePrice'])
plt.scatter(train['GarageCars'], train['SalePrice'])