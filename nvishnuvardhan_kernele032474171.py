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
train_set = pd.read_csv("/kaggle/input/train.csv")

test_set = pd.read_csv("/kaggle/input/test.csv")

test_y1 = pd.read_csv("/kaggle/input/sample_submission.csv")

test_y = test_y1["SalePrice"]

print(train_set.shape)

print(test_set.shape)
train_set[['SaleType']].shape

train_set['SaleType']

XX = train_set.select_dtypes(include=[object])

categorical_feature_mask = train_set.dtypes==object

categorical_feature_mask = train_set.columns[categorical_feature_mask].tolist();

categorical_feature_mask
train_set['SalePrice']

corr_matrix =train_set.corr()

corr_matrix['SalePrice'].sort_values(ascending=False)



needed_columns = ["OverallQual",'GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd',

            'GarageYrBlt','MasVnrArea','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch']

X = train_set[needed_columns]

y = train_set['SalePrice']

print(X)
null_columns=X.columns[X.isnull().any()]

X[null_columns].isnull().sum()
median = int(X['MasVnrArea'].median())

X['MasVnrArea'].fillna(int(median),inplace=True)

median = X['LotFrontage'].median()

X['LotFrontage'].fillna(int(median),inplace=True)

median = X['GarageYrBlt'].median()

X['GarageYrBlt'].fillna(int(median),inplace=True)

print(X.shape)

print(X.shape)

test_X  = test_set[needed_columns]

print(test_X.shape)

null_columns=X.columns[X.isnull().any()]

X[null_columns].isnull().sum()

null_columns=test_X.columns[test_X.isnull().any()]

test_X[null_columns].isnull().sum()
median = test_X['MasVnrArea'].median()

test_X['MasVnrArea'].fillna(int(median),inplace=True)

median = test_X['BsmtFinSF1'].median()

test_X['BsmtFinSF1'].fillna(median,inplace=True)

median = test_X['LotFrontage'].median()

test_X['LotFrontage'].fillna(int(median),inplace=True)

median = test_X['BsmtFullBath'].median()

test_X['BsmtFullBath'].fillna(median,inplace=True)

median = test_X['BsmtUnfSF'].median()

test_X['BsmtUnfSF'].fillna(median,inplace=True)

median = test_X['GarageYrBlt'].median()

test_X['GarageYrBlt'].fillna(int(median),inplace=True)

median = test_X['GarageCars'].median()

test_X['GarageCars'].fillna(median,inplace=True)

median = test_X['GarageArea'].median()

test_X['GarageArea'].fillna(median,inplace=True)

median = test_X['TotalBsmtSF'].median()

test_X['TotalBsmtSF'].fillna(median,inplace=True)

null_columns=test_X.columns[test_X.isnull().any()]

test_X[null_columns].isnull().sum()
null_columns=X.columns[X.isnull().any()]

X[null_columns].isnull().sum()

null_columns=test_X.columns[test_X.isnull().any()]

test_X[null_columns].isnull().sum()
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()

X_t = pd.DataFrame(scaler.fit_transform(X));

X_te = pd.DataFrame(scaler.fit_transform(test_X));

# print(X[:5])

# print(X_t[:5])

type(X_t)



maxprice = train_set['SalePrice'].max()

y = y/maxprice

test_y = test_y/maxprice

from tensorflow import keras

from keras.models import Sequential

# model  = keras.Sequential()

# model

# model.add(keras.layers.Input(shape=X.shape[1:]));

# model.add(keras.layers.Dense(20, activation="relu"));

# model.add(keras.layers.Dense(10, activation="relu"));

# model.add(keras.layers.Dense(1));

# input1 = keras.layers.Input(shape=X.shape[1:])

# hidden1 = keras.layers.Dense(300, activation="relu")(input1)

# hidden2 = keras.layers.Dense(300, activation="relu")(hidden1)

# hidden3  = keras.layers.Dense(100, activation="relu")(hidden2)

# output = keras.layers.Dense(1)(hidden3)

from keras.models import Sequential

from keras import regularizers

from keras.layers import Dense

model = Sequential()

model.add(Dense(300, activation='relu',input_dim=X.shape[1:][0]))

model.add(Dense(100,activation='relu'))

model.add(Dense(50,activation='relu'))

model.add(Dense(10,activation='relu'))

model.add(Dense(output_dim=1,activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])





history=model.fit(X_t,y,epochs=30)

predictions = model.predict(X_te)

print(predictions[:5],"  ", test_y[:5])

#history.history
print(predictions[:5]*maxprice,"  ", test_y[:5]*maxprice)
submission = pd.read_csv('../input/test.csv')

final = submission[['Id']].copy()

predictions = predictions * maxprice

final['SalePrice'] = predictions

del submission





final.to_csv('submission.csv',index=False)