# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Watching data
path = '../input/Housing.csv'
data = pd.read_csv(path)

data.head()
# Visualization of data
plt.scatter(data['lotsize'], data['price'], color='red')
plt.show()
# preparation for training
Y = data['price']
X = data.drop('price', axis=1)
X = X.drop('Unnamed: 0', axis=1)

X.head()
# Check missing values
data.isnull().sum()
# Apply one-hot-encoding
X = pd.get_dummies(X)
print(X.dtypes.sample(17))
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
model = DecisionTreeRegressor(max_depth=10)
model.fit(train_X, train_Y)
predicted_value = model.predict(test_X)
print('Algorithm is DecisionTreeRegressor')
print('Mean absolute error: ' + str(mean_absolute_error(test_Y, predicted_value)))

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=10)
model.fit(train_X, train_Y)
predicted_value = model.predict(test_X)
print('Algorithm is RandomForestRegressor')
print('Mean absolute error: ' + str(mean_absolute_error(test_Y, predicted_value)))
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_X, train_Y)
predicted_value = model.predict(test_X)
print('Algorithm is LinearRegression')
print('Mean absolute error: ' + str(mean_absolute_error(test_Y, predicted_value)))