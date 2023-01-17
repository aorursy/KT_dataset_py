import numpy as np

import pandas as pd

import statsmodels.api as sm

import seaborn as sns

import matplotlib.pyplot as plt

import math
data = pd.read_csv("../input/home-data-for-ml-course/train.csv")
data.head()
data.describe()
corr = data.corr()

corr.style.background_gradient(cmap='coolwarm')
endog = data['SalePrice']

exog = sm.add_constant(data[['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']])

print(exog)
X = exog.to_numpy()

y = endog.to_numpy()

xt = np.transpose(X)

print(xt)
xt_X = np.matmul(xt,X)

print(xt_X)
xt_X_inv = np.linalg.inv(xt_X)

print(xt_X_inv)
xt_X_inv_xt = np.matmul(xt_X_inv,xt)

print(xt_X_inv_xt)
beta = np.matmul(xt_X_inv_xt,y)

print(beta)
mod = sm.OLS(endog,exog)

results = mod.fit()

print(results.summary())
def RSE(y_true, y_predicted):

   

    y_true = np.array(y_true)

    y_predicted = np.array(y_predicted)

    RSS = np.sum(np.square(y_true - y_predicted))



    rse = math.sqrt(RSS / (len(y_true) - 2))

    return rse
yp= results.predict()

ypa = np.array(yp)

yta = data['SalePrice']

eterms =yta-ypa





data1 = pd.DataFrame(eterms)

data1['SalePrice'].hist(bins=10)
rse= RSE(data['SalePrice'],results.predict())

print(rse)
from sklearn import linear_model

X = data[['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']]

y = data['SalePrice']



lm = linear_model.LinearRegression()

model = lm.fit(X,y)

lm.coef_
lm.intercept_
test = pd.read_csv('../input/home-data-for-ml-course/test.csv')
X_test = data[['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']]

y_pred = lm.predict(X_test)
y_pred