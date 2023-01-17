# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import xgboost as xgb

from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

import os

root_dir = os.path.join("../input")

print(os.listdir(root_dir))



# Any results you write to the current directory are saved as output.
df_train = pd.read_json(os.path.join(root_dir, 'train.json'))

df_train.head()
df_test = pd.read_json(os.path.join(root_dir, 'test.json'))

df_test.head()
X_train = np.array(df_train.iloc[:-1000]['vibrates'].tolist())

X_train = X_train.reshape(len(X_train), -1)

Y_train = df_train.iloc[:-1000]['seismic_intensity'].values



X_valid = np.array(df_train.iloc[-1000:]['vibrates'].tolist())

X_valid = X_valid.reshape(len(X_valid), -1)

Y_valid = df_train.iloc[-1000:]['seismic_intensity'].values



X_test = np.array(df_test['vibrates'].tolist())

X_test = X_test.reshape(len(X_test), -1)



print(X_train.shape)

print(Y_train.shape)

print(X_valid.shape)

print(Y_valid.shape)

print(X_test.shape)
# Set our parameters for xgboost

params = {}

params['objective'] = 'reg:squarederror'

params['eval_metric'] = 'mae'

params['eta'] = 0.01

params['max_depth'] = 5



d_train = xgb.DMatrix(X_train, label=Y_train)

d_valid = xgb.DMatrix(X_valid, label=Y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 300, watchlist, early_stopping_rounds=100, verbose_eval=10)

Y_pred = bst.predict(xgb.DMatrix(X_valid))



# eval

print('The mae of prediction is:', mean_absolute_error(Y_valid, Y_pred))

Y_pred = bst.predict(xgb.DMatrix(X_test))

df_test['seismic_intensity'] = Y_pred

df_test.head()
df_test[['index', 'seismic_intensity']].to_csv('submit.csv', index=False)