import os

import pandas as pd 

import numpy as np 

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder



print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

sample= pd.read_csv('../input/sample_submission.csv')
train.shape , test.shape ,sample.shape
plt.figure(figsize=(30,15))

sns.heatmap(train.corr(), annot=True,annot_kws={"size": 9}, cmap='inferno')

plt.show()

train_X = train[['SalePrice','OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','GarageCars', 'GarageArea','TotRmsAbvGrd']].copy()
plt.figure(figsize=(25,15))

sns.heatmap(train_X.corr(), annot=True,annot_kws={"size": 15}, cmap='inferno')
sns.pairplot(train_X)
train_X.describe()
sns.boxplot(train_X['SalePrice'])
plt.scatter(train_X['SalePrice'],train_X['OverallQual'])


test_X=test[['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','GarageCars', 'GarageArea','TotRmsAbvGrd']].copy()

test_X.info()
X = train_X.drop('SalePrice',axis=1)

y = train_X['SalePrice']
X.shape , y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape,y_train.shape,X_test.shape,y_test.shape
from sklearn.linear_model import LinearRegression

model =LinearRegression()

model.fit(X,y)
x = model.score(X,y)

x
test_X['TotalBsmtSF'] = test_X['TotalBsmtSF'].fillna(-999)

test_X['GarageCars'] = test_X['GarageCars'].fillna(-999)

test_X['GarageArea'] = test_X['GarageArea'].fillna(-999)

test_X.isna().sum()

y_predict  = model.predict(test_X)
from xgboost import XGBRegressor

XGB = XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=1000,reg_alpha=0.001,reg_lambda=0.000001,n_jobs=-1,min_child_weight=3)

XGB.fit(X_train,y_train)
y_predict1 = print ("Training score:",XGB.score(X_train,y_train),"Test Score:",XGB.score(X_test,y_test))
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_predict1})

my_submission.to_csv('submission3.csv', index=False)