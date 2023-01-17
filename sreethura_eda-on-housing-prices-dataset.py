#Importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn.preprocessing as Standardscaler

%matplotlib inline
train = pd.read_csv("C:\\Users\\sreeram\\Desktop\\pythonfiles\\House-Train.csv")

train.head()
print(train.dtypes)

train.columns
train['SalePrice'].describe()
#Correlations with the target variable

a = train.corr()

b = print(a['SalePrice'].sort_values(ascending = False )[:10],'\n')
#Histogram

sns.distplot(train['SalePrice'])
print("Skewness: %f" % train['SalePrice'].skew())

print("kurtosis: %f" % train['SalePrice'].kurt())
var = 'GrLivArea'

plt= pd.concat([train['SalePrice'], train[var]], axis=1)

plt.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
var = 'TotalBsmtSF'

plt= pd.concat([train['SalePrice'], train[var]], axis=1)

plt.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([train['SalePrice'],train[var]], axis = 1)

f,ax = plt.subplots(figsize = (8,6))

fig = sns.boxplot(x=var, y='SalePrice',data = data)

fig.axis(ymin =0 ,ymax = 800000)
#Missing data

missing = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([missing,percent],axis = 1 ,keys=['missing','percent'])

missing_data.head(20)
#Pair Plot 

sns.set()

cols = ['SalePrice','GrLivArea'  ,'GarageCars' , 'GarageArea' ,'TotalBsmtSF' ,'1stFlrSF','FullBath','YearBuilt']

sns.pairplot(train[cols],size = 2.5)

plt.show()