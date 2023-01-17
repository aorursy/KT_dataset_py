import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import sklearn

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import cross_val_score
kaggle_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') 

kaggle_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

da=pd.concat([kaggle_train,kaggle_test]).reset_index()
noSalePrice = [a for a in da.columns if a!='SalePrice']

da[noSalePrice] = da[noSalePrice].fillna(da[noSalePrice].mean())
da.plot.scatter(x='GrLivArea',y='SalePrice',title='SalePrice Vs GrLivArea',figsize=(3,3));

da.plot.scatter(x='OverallQual',y='SalePrice',title='SalePrice Vs OverallQual',figsize=(3,3));

da.plot.scatter(x='TotalBsmtSF',y='SalePrice',title='SalePrice Vs TotalBsmtSF',figsize=(3,3));
da.drop(da[(da['SalePrice']<250000)&(da['GrLivArea']>4000)].index,inplace=True)

da.drop(da[(da['SalePrice']<500000)&(da['TotalBsmtSF']>3000)].index,inplace=True)

da.drop(da[(da['SalePrice']>700000)&(da['TotalBsmtSF']>2000)].index,inplace=True)

da.drop(da[(da['SalePrice']>200000)&(da['OverallQual']==4)].index,inplace=True)

da.drop(da[(da['SalePrice']>500000)&(da['OverallQual']==8)].index,inplace=True)
da.plot.scatter(x='GrLivArea',y='SalePrice',title='SalePrice Vs GrLivArea',figsize=(3,3));

da.plot.scatter(x='OverallQual',y='SalePrice',title='SalePrice Vs OverallQual',figsize=(3,3));

da.plot.scatter(x='TotalBsmtSF',y='SalePrice',title='SalePrice Vs TotalBsmtSF',figsize=(3,3));
ind_vars=['OverallCond','GrLivArea', 'TotalBsmtSF','GarageArea','YearBuilt','OverallQual','GarageCars']
from patsy import dmatrices

Y, X = dmatrices('SalePrice ~ GrLivArea+TotalBsmtSF+OverallQual+GarageArea+YearBuilt+OverallCond+GarageCars', da, return_type='dataframe')

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif
dac = da[['SalePrice','OverallCond','GrLivArea', 'TotalBsmtSF','GarageArea','YearBuilt','OverallQual','GarageCars']].corr()

dac[['SalePrice']][1:]
da['SalePrice'] = np.log1p(da['SalePrice'])
Xtrain,Xtest,Ytrain,Ytest=train_test_split(da[:1452][ind_vars],da[:1452]['SalePrice'],test_size=0.33,random_state=42)
model=LinearRegression()

model.fit(Xtrain,Ytrain)

Ypred1=model.predict(Xtest)

dff=pd.DataFrame({"Predicted Values":np.expm1(Ypred1),"     Values should have been predicted":np.expm1(Ytest.tolist())})

print(dff.head())
def adjusted_r2(Xtest,r2):

  n=len(Xtest)

  p=len(Xtest.columns)

  a_r2=1 - (1-r2)*(n-1)/(n-p-1)

  return a_r2

R2_1=r2_score(Ytest,Ypred1)

adj_r2_1=adjusted_r2(Xtest,R2_1)

print("R2 value : {}".format(R2_1))

print("Adjusted R2 value : {}".format(adj_r2_1))

print("MSE : {}".format(sqrt(mean_squared_error(Ytest,Ypred1))))
def adjusted_r2(Xtest,r2):

  n=len(Xtest)

  p=len(Xtest.columns)

  a_r2=1 - (1-r2)*(n-1)/(n-p-1)

  return a_r2

Ls1=LassoCV()

Ls1.fit(Xtrain,Ytrain)

Ypred2=Ls1.predict(Xtest)

R2_2=r2_score(Ytest,Ypred2)

adj_r2_2=adjusted_r2(Xtest,R2_2)

print("R2 value : {}".format(R2_2))

print("Adjusted R2 value : {}".format(adj_r2_2))

print("Error : {}".format(sqrt(mean_squared_error(Ytest,Ypred2))))
Rr=RidgeCV()

Rr.fit(Xtrain,Ytrain)

Ypred3=Rr.predict(Xtest)

R2_3=r2_score(Ytest,Ypred3)

adj_r2_3=adjusted_r2(Xtest,R2_3)

print("R2 value : {}".format(R2_3))

print("Adjusted R2 value : {}".format(adj_r2_3))

print("Error : {}".format(sqrt(mean_squared_error(Ytest,Ypred3))))
EN1 = ElasticNetCV(l1_ratio=np.linspace(0.1, 1.0, 5))

EN1.fit(Xtrain,Ytrain)

Ypred4=EN1.predict(Xtest)

R2_4=r2_score(Ytest,Ypred4)

adj_r2_4=adjusted_r2(Xtest,R2_4)

print("R2 value : {}".format(R2_4))

print("Adjusted R2 value : {}".format(adj_r2_4))

print("Error : {}".format(sqrt(mean_squared_error(Ytest,Ypred4))))
x=da[1452:].drop(['SalePrice'],axis=1)

sub=pd.DataFrame({'Id':x.Id,'SalePrice':np.expm1(Rr.predict(x[ind_vars]))})

sub.to_csv('final_submission.csv',index=False)