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
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
"train: ", train.shape, "test:",  test.shape
train.head()
test.head()
train_test = pd.concat([train.drop(['SalePrice'], axis=1), test], axis=0, ignore_index=True)
train_test.head()
train_test.tail()
train_test.info()
train_test_float = train_test.select_dtypes(exclude='object')
train_test_float.head()
def nan_cols():

    total = train_test_float.isna().sum().sort_values(ascending=False)

    percent = (train_test_float.isna().sum() / train_test_float.isna().count()).sort_values(ascending=False)



    nan_cols = pd.concat([total, percent], axis=1, keys=['Total', '%'])



    print(nan_cols[nan_cols['%'] > 0])
nan_cols()
import seaborn as sns

import matplotlib.pyplot as plt



train_corr = train.corr()



column = 'SalePrice'

corr_cols_n = 10



cols = train_corr.nlargest(corr_cols_n, column)[column].index    

coef = train_corr.nlargest(corr_cols_n, column)[cols].values



plt.figure(figsize=(10, 10))



sns.heatmap(coef, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=cols.values, xticklabels=cols.values)
train_test_float.drop(['Id', 'LotFrontage', 'GarageYrBlt', 'MasVnrArea'], axis=1, inplace=True)
train_test_float.head()
nan_cols()
# train_test_float.fillna(train_test_float['MasVnrArea'].mean(), inplace=True)

train_test_float.fillna(train_test_float['BsmtHalfBath'].median(), inplace=True)

train_test_float.fillna(train_test_float['BsmtFullBath'].median(), inplace=True)

train_test_float.fillna(train_test_float['GarageArea'].mean(), inplace=True)

train_test_float.fillna(train_test_float['BsmtFinSF1'].mean(), inplace=True)

train_test_float.fillna(train_test_float['BsmtFinSF2'].mean(), inplace=True)

train_test_float.fillna(train_test_float['BsmtUnfSF'].mean(), inplace=True)

train_test_float.fillna(train_test_float['TotalBsmtSF'].mean(), inplace=True)

train_test_float.fillna(train_test_float['GarageCars'].mean(), inplace=True)
nan_cols()
outliar_rows = set()

train_test_float.head()
train_test_float.columns
sns.scatterplot(y='LotArea', x=train.index, data=train)
train[train['LotArea'] > 100000]
[outliar_rows.add(i) for i in list(train[train['LotArea'] > 100000].index)]

outliar_rows
sns.scatterplot(y='MasVnrArea', x=train.index, data=train)
# train[train['MasVnrArea'] > 1300] 
# [outliar_rows.add(i) for i in list(train[train['MasVnrArea'] > 1300].index)]

# outliar_rows
sns.scatterplot(y='GarageArea', x=train.index, data=train)
train[train['GarageArea'] > 1200]
[outliar_rows.add(i) for i in list(train[train['GarageArea'] > 1200].index)]

outliar_rows
train_test_float.describe()
sns.scatterplot(y='MiscVal', x=train.index, data=train)
[outliar_rows.add(i) for i in list(train[train['MiscVal'] > 6000].index)]

outliar_rows
sns.scatterplot(y='PoolArea', x=train.index, data=train)
[outliar_rows.add(i) for i in list(train[train['PoolArea'] > 400].index)]

outliar_rows
len(outliar_rows)
from sklearn.preprocessing import MinMaxScaler



X = train_test_float.values

y = train.drop(list(outliar_rows), axis=0)['SalePrice'].values

y = y.reshape(y.shape[0], 1)



# scaling X values

scaler = MinMaxScaler()

scaledX = scaler.fit_transform(X)



# scaling y values

scaler_y = MinMaxScaler()

y_scaled = scaler_y.fit_transform(y)



train_test_float_scaled = pd.DataFrame(scaledX, columns=train_test_float.columns)
"train_test_float_scaled", train_test_float_scaled.shape, "y_scaled", y_scaled.shape
"train: ", train.shape, "test: ", test.shape
train_int = train_test_float_scaled[:train.shape[0]]

train_int.shape
test_int = train_test_float_scaled[test.shape[0]+1:]

test_int.shape
train_int.head()
train_int.drop(list(outliar_rows), axis=0, inplace=True)
## Train, test split



from sklearn.model_selection import train_test_split



X = train_int.values



x_train, x_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2)



print("x_train.shape: {}".format(x_train.shape))

print("y_train.shape: {}".format(y_train.shape))

print()

print("x_test.shape: {}".format(x_test.shape))

print("y_test.shape: {}".format(y_test.shape))
from tensorflow.keras import Sequential

from tensorflow.keras import layers

from tensorflow.keras import optimizers



model = Sequential()



model.add(layers.Dense(64, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))

model.add(layers.Dense(128, kernel_initializer='normal',  activation='relu'))

model.add(layers.Dense(256, kernel_initializer='normal',  activation='relu'))

model.add(layers.Dense(128, kernel_initializer='normal', activation='relu'))

model.add(layers.Dense(64, kernel_initializer='normal'))

model.add(layers.Dense(1, kernel_initializer='normal'))



model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100, batch_size=10)
model.evaluate(x_test, y_test)
test_int.head()
test_int.shape
X = test_int.values
submission_pred = model.predict(X)
submission_pred_inversed = scaler_y.inverse_transform(submission_pred)
test_int.index
submission = pd.DataFrame()



submission['Id'] = test_int.index + 1

submission['SalePrice'] = submission_pred_inversed



submission.head()
submission.to_csv('submission.csv', index=False)