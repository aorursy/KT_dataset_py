import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

tr=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

ts=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

ss=pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")



tr.head()

tr.columns



xtrain=tr.iloc[:,[1,5,79,78,77,17,18,19]]

ytrain=tr.loc[:,['SalePrice']]

xtest=ts.iloc[:,[1,5,79,78,77,17,18,19]]

xtrain.head()

xtest=xtest.fillna('SaleType')

xtest.info()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

xtrain.iloc[:,1]=le.fit_transform(xtrain.iloc[:,1])

xtrain.iloc[:,3]=le.fit_transform(xtrain.iloc[:,3])



xtrain.iloc[:,2]=le.fit_transform(xtrain.iloc[:,2])



xtrain





xtest.iloc[:,1]=le.fit_transform(xtest.iloc[:,1])

xtest.iloc[:,2]=le.fit_transform(xtest.iloc[:,2])

xtest.iloc[:,3]=le.fit_transform(xtest.iloc[:,3])

xtest

xtrain

ytrain
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(xtrain,ytrain)

pred=lr.predict(xtest)

pred

from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor(max_depth=3,random_state=0)

dt.fit(xtrain,ytrain)

pred=dt.predict(xtest)

pred

from sklearn.ensemble import RandomForestRegressor

dt1=RandomForestRegressor (max_depth=3,random_state=0)

dt1.fit(xtrain,ytrain)

pred=dt1.predict(xtest)

pred
