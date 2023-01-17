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
df_train = pd.read_csv(
    '../input/sales_train.csv',
    usecols = [1, 2, 3, 5]
).fillna(0)
print (df_train.shape)
print (df_train.info())
df_t = df_train.pivot_table(
    values='item_cnt_day',
    index=['shop_id', 'item_id'],
    columns='date_block_num',
    aggfunc='sum'
).fillna(0)
print (df_t.head())
print (df_t.shape)
X_train = pd.DataFrame()
y_train = pd.Series()
X_val = pd.DataFrame()
y_val = pd.Series()
for i in range(2, 32):
    print (i)
    print ('='*50)
    X = df_t.iloc[:, i-2:i]
    X.columns = ['P-2', 'P-1']
    X_train = X_train.append(X)
    y = pd.Series(df_t.iloc[:, i])
    y_train = y_train.append(y)
for j in range(32, 34):
    print (j)
    print ('='*50)
    X = df_t.iloc[:, i-2:i]
    X.columns = ['P-2', 'P-1']
    X_val = X_val.append(X)
    y = df_t.iloc[:, i]
    y_val = y_val.append(y)
print (len(X_train), len(y_train))
print (len(X_val), len(y_val))
X_train2 = X_train.as_matrix()
print (X_train2)
X_train2 = X_train2.reshape(X_train2.shape[0], 1, X_train2.shape[1])
y_train2 = y_train.values
print (y_train2)
X_val2 = X_val.as_matrix()
X_val2 = X_val2.reshape(X_val2.shape[0], 1, X_val2.shape[1])
y_val2 = y_val.values
print (X_val2)
print (y_val2)
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(64, input_shape=(X_train2.shape[1],X_train2.shape[2])))
model.add(Dropout(.1))
model.add(Dense(32))
model.add(Dropout(.2))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(X_train2, y_train2, batch_size = 100000, epochs = 20, verbose=2,
    validation_data=(X_val2,y_val2))
df_test = pd.read_csv(
        '../input/test.csv',
        ).set_index(['shop_id', 'item_id'])
X_test = df_test
X = df_t.iloc[:, 32:34]
X_test = X_test.join(X)
print (X_test)
X_test = X_test.fillna(0)
print (X_test)
X_test2 = X_test.iloc[:, 1:3]
X_test2 = X_test2.as_matrix()
print (X_test2.shape)
X_test2 = X_test2.reshape(X_test2.shape[0], 1, X_test2.shape[1])
predict = model.predict(X_test2)
df_test['item_cnt_month'] = predict
df_test.to_csv('accountant_keras1.csv', float_format = '%.2f', index=None)