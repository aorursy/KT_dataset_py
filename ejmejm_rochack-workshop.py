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
df_train = pd.read_csv('../input/train.csv')
df_train.head()
df_train.columns
df_train.describe()
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
X_cols = ['YearBuilt', 'GrLivArea', 'TotRmsAbvGrd', 'GarageArea', 'LotArea']
cols_w_tgt = ['YearBuilt', 'GrLivArea', 'TotRmsAbvGrd', 'GarageArea', 'LotArea', 'SalePrice']

sns.pairplot(df_train[cols_w_tgt], diag_kind='kde')
fig = plt.figure(figsize=(30, 8))

sns.boxplot(x='YearBuilt', y='SalePrice', data=df_train[df_train['YearBuilt'] >= 1970])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_data = df_train[X_cols]
y_data = df_train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, shuffle=True)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train);
print(lr_model.predict(X_test[:20]))
preds = lr_model.predict(X_test[:20])
actual = y_test[:20]

for p, a in zip(preds, actual):
    print('Prediction: {} \t Actual: {}'.format(int(p), int(a)))
from sklearn.metrics import mean_squared_error
from math import sqrt
preds = lr_model.predict(X_test)
actual = y_test.values

RMSE = sqrt(mean_squared_error(actual, preds))
print('RMSE:', RMSE)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBRegressor
models = {'Linear Regression': LinearRegression(),
          'Random Forest': RandomForestRegressor(),
          'SVC': SVC(),
          'XGBoost': XGBRegressor()}

actual = y_test.values

model_rmse = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    RMSE = sqrt(mean_squared_error(actual, preds))
    model_rmse[model_name] = RMSE
for key in model_rmse:
    print('{} RMSE: {}'.format(key, model_rmse[key]))
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(32, input_dim=5, activation='relu', bias_initializer='random_uniform'))
model.add(Dense(64, activation='relu', bias_initializer='random_uniform'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam') #metrics=['accuracy'])
X_data = df_train[X_cols]
y_data = df_train['SalePrice']

for col in X_data.columns:
    X_data[col] = X_data[col] / max(X_data[col])
    
y_max = max(y_data)
y_data = y_data / y_max
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, shuffle=True)
model.fit(X_train, y_train, batch_size=64, epochs=10)
preds = model.predict(X_test[:20])
actual = y_test[:20]
for p, a in zip(preds, actual):
    print('Prediction: {} \t Actual: {}'.format(int(p*y_max), int(a*y_max)))
preds = model.predict(X_test) * y_max
actual = y_test * y_max

RMSE = sqrt(mean_squared_error(actual, preds))
print('RMSE:', RMSE)