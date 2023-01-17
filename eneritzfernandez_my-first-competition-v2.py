# Loading some useful libraries



import numpy as np 

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing train and test data



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.describe()
#Name of each of the colums in the train data set



train.columns
#Define the y variable (price)



Y_train = train.SalePrice
Y_train.head()
train.describe()
train.info()
features = ['MSSubClass', 'LotArea', 

        'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',  '1stFlrSF', 'YrSold', 'OpenPorchSF' , 'MiscVal', 'BedroomAbvGr', 'TotRmsAbvGrd', 

            'TotalBsmtSF', 'GrLivArea']# 'EnclosedPorch']# ,  '3SsnPorch', 'ScreenPorch'  ]
from sklearn.model_selection import train_test_split
X_train = train[features]



train_X, val_X, train_y, val_y = train_test_split(X_train, Y_train, random_state = 0)

X_train.head()
train_X.info()
#from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



# Define model. Specify a number for random_state to ensure same results each run

model = XGBRegressor(n_estimators=300, learning_rate=0.15)



# Fit model

model.fit(train_X, train_y)
Y_train_prediction = model.predict(train_X)
from sklearn.metrics import mean_absolute_error



mae1 = mean_absolute_error(train_y, Y_train_prediction)



Y_val_prediction = model.predict(val_X)



mae2 = mean_absolute_error(val_y, Y_val_prediction)



print(mae1)

print(mae2)
test.info()
X_test = test[features]



Y_test = model.predict(X_test)



print(Y_test)
my_submission = test.Id
my_submission.head()
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': Y_test})

my_submission.to_csv('submission.csv', index=False)



print(my_submission.head())