#importing python libraries for loading data and data exploration

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')
train.head()
test = pd.read_csv('../input/test.csv')
test.head()
train.shape
train.info()
train.drop(['PoolQC','MiscFeature','Fence','FireplaceQu', 'Alley'],axis=1,inplace=True)

test.drop(['PoolQC','MiscFeature','Fence','FireplaceQu', 'Alley'],axis=1,inplace=True)
# dropping some additional columns will need to check it again

train.drop(['Condition2','Electrical','Exterior1st','Exterior2nd','GarageQual','Heating','HouseStyle','RoofMatl','Utilities'],axis=1,inplace=True)

test.drop(['Condition2','Electrical','Exterior1st','Exterior2nd','GarageQual','Heating','HouseStyle','RoofMatl','Utilities'],axis=1,inplace=True)
train.LotFrontage.fillna(train.LotFrontage.mean(),inplace=True)

train.MasVnrArea.fillna(train.MasVnrArea.mean(),inplace=True)
train.fillna(method='bfill',inplace=True)
test.fillna(method='bfill',inplace=True)
train = pd.get_dummies(train)
train.head()
test = pd.get_dummies(test)
test.head()

test_id = test.Id
train.columns ^ test.columns
from sklearn.model_selection import train_test_split
X = train.drop('SalePrice',axis=1)

y = train.SalePrice
X_train,X_test,y_train,y_test = train_test_split(X,y)
X_train.head()
# linear regression

from sklearn.linear_model import LinearRegression

linReg = LinearRegression(normalize=False)

linReg.fit(X_train, y_train)

linReg.score(X_test, y_test)
# scaling values

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

#y_train = sc.fit_transform(y_train)

test = sc.fit_transform(test)

X_test = sc.fit_transform(X_test)

#y_test = sc.fit_transform(y_test)
# Gradient Boosting regression

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=800,learning_rate=0.1,max_depth=3, random_state=0)

gbr.fit(X_train, y_train)
gbr.score(X_test, y_test)
y_predict = gbr.predict(test)
submission = pd.DataFrame({'Id':test_id,'SalePrice':y_predict})
submission.to_csv('submission.csv',index=False)