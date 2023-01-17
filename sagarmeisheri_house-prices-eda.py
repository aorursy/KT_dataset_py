import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets, linear_model

import seaborn as sns

import datetime as dt

import pylab 

import scipy.stats as stats

import statsmodels.api as sm



%matplotlib inline



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()

test.head()
#Recode certain columns to categorical since they have numeric values

train.loc[:,['MSSubClass','OverallQual','OverallCond']] = train.loc[:,['MSSubClass','OverallQual','OverallCond']].astype(str)

test.loc[:,['MSSubClass','OverallQual','OverallCond']] = test.loc[:,['MSSubClass','OverallQual','OverallCond']].astype(str)
#get numeric columns 

num_col_train = list(train.select_dtypes(include=[np.number]).columns)

categ_col_train = train.columns.difference(num_col_train)



num_col_test = list(test.select_dtypes(include=[np.number]).columns)

categ_col_test = test.columns.difference(num_col_train)
# Missing Values #



# Find columns with missing values

print(train.isnull().sum().sort_values(ascending = False))

print(test.isnull().sum().sort_values(ascending = False))



#Fill numeric missing values as 0

train[categ_col_train] = train[categ_col_train].fillna(value = 'None')

test[categ_col_test] = test[categ_col_test].fillna(value = 'None')



#Fill categorical missing values as None

train[num_col_train] = train[num_col_train].fillna(value = 0)

test[num_col_test] = test[num_col_test].fillna(value = 0)
#Exploratory Analysis



#Uni-variate Distribution



print(train[num_col_train].describe())



#sns.distplot(train[['SalePrice']])

print(stats.describe(train[['SalePrice']]))

print(stats.describe(train[['LotArea']]))



train[['SalePrice']] = np.log(train[['SalePrice']])

train[['LotArea']] = np.log(train[['LotArea']])

test[['LotArea']] = np.log(test[['LotArea']])
#Scatterplots & Multi-variate Distributions

var2 = ['LotArea','YearBuilt','YrSold','TotalBsmtSF','GarageArea','SalePrice']

sns.jointplot(x = "SalePrice", y = "YearBuilt", data = train)

sns.pairplot(train[var2])
#Catgeorical Data



train[categ_col_train].describe(include = "all")



categ = train['Neighborhood'].unique()

g = sns.boxplot(x = "Neighborhood", y = "SalePrice", data = train)

g.set_xticklabels(categ, rotation=90)



sns.barplot(x = "MSSubClass", y = "SalePrice",data = train)



sns.countplot(x = "MSZoning",data = train)