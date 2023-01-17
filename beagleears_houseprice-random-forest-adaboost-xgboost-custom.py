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
# Load the training data

df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt
X=df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']]

y=df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestRegressor
rnd_reg = RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)

rnd_reg.fit(X_train, y_train)

y_pred_rf = rnd_reg.predict(X_test)

print("RandomForestRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_rf)))
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=200, learning_rate=0.5)

ada_reg.fit(X_train, y_train)

y_pred_ada=ada_reg.predict(X_test)
print("AdaBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_ada)))
from xgboost import XGBRegressor
xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)

y_pred_xgb = xgb_reg.predict(X_test)
print("XGBRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_xgb)))
# Imported a fresh copy

df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")



# Get the names of all the columns, then remove the target

names = list(df.columns)

names.remove('SalePrice')



# List of features needing one-hot encoding, with corresponding column names post-encoding

one_hot_list = ['Alley', 'GarageType', 'GarageFinish', 'GarageQual', 'MSZoning', 'SaleCondition', 'Street']

feature_names = []



def one_hot(name):

    '''Trying to automate some one-hot encoding'''

    global df

    global feature_names

    

    temp_df = pd.get_dummies(df[name], prefix=name)

    [feature_names.append(c) for c in list(temp_df.columns)]

    

    df = pd.concat([df, temp_df], axis=1)

    df.drop([name], axis=1, inplace=True)





# Apply the one_hot function to every column in the one_hot_list

[one_hot(name) for name in one_hot_list]
# Feature Engineering - TotalBaths

bath_columns = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']



# Create a new column and set each value equal to zero

df['TotalBaths'] = 0



df['TotalBaths'] = df.apply(lambda x: df['BsmtFullBath'] + df['BsmtHalfBath'] + df['FullBath'] + df['HalfBath'])



# Add the new column name to the list of features we want to use

feature_names.append('TotalBaths')
features = feature_names + ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF']

features.remove('GarageQual_Ex')



target = ['SalePrice']
X=df[features]

y=df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)

y_pred_xgb = xgb_reg.predict(X_test)
print("XGBRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_xgb)))
df2 = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



# Get the names of all the columns, then remove the target

names = list(df2.columns)



# List of features needing one-hot encoding, with corresponding column names post-encoding

one_hot_list = ['Alley', 'GarageType', 'GarageFinish', 'GarageQual', 'MSZoning', 'SaleCondition', 'Street']

feature_names = []



def one_hot(name):

    '''Trying to automate some one-hot encoding'''

    global df2

    global feature_names

    

    temp_df = pd.get_dummies(df2[name], prefix=name)

    [feature_names.append(c) for c in list(temp_df.columns)]

    

    df2 = pd.concat([df2, temp_df], axis=1)

    df2.drop([name], axis=1, inplace=True)





# Apply the one_hot function to every column in the one_hot_list

[one_hot(name) for name in one_hot_list]

# Feature Engineering - TotalBaths

bath_columns = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']



# Create a new column and set each value equal to zero

df2['TotalBaths'] = 0



df2['TotalBaths'] = df2.apply(lambda x: df2['BsmtFullBath'] + df2['BsmtHalfBath'] + df2['FullBath'] + df2['HalfBath'])



# Add the new column name to the list of features we want to use

feature_names.append('TotalBaths')
[col for col in df.columns]
[col for col in df2.columns]
test_X=df2[features]

test_y=test_X.values.reshape(-1,len(features))

predicted_price=xgb_reg.predict(test_X)
my_submission = pd.DataFrame({'Id': df2.Id, 'SalePrice': predicted_price})

my_submission.to_csv('submissionXGBoost.csv', index=False)
! ls