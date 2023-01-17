# Libraries

import numpy as np 

import pandas as pd 

import matplotlib.pylab as plt

import seaborn as sns



train = pd.read_csv("../input/train.csv")
train.head()
train.describe()
train.info()
y = train["SalePrice"].reshape(-1,1)

y.shape
train["PoolQC"].value_counts()
train["MiscFeature"].value_counts()
train["FireplaceQu"].value_counts()
X = train.loc[:,"LotArea"].reshape(-1,1)
X.info()
from sklearn import linear_model
temp = train.dropna(axis=1) # Remove data with NA, model will not accept them

temp = temp.loc[:,"MSSubClass":"SaleCondition"] # Remove id and price data

X = pd.DataFrame()

for col in temp:

    column = temp[col]

    if column.dtype == np.int64:

        X[col] = column
X.info()
type(temp)
model = linear_model.LinearRegression()

model.fit(X,y)

model.coef_
model.score(X,y)
import statsmodels.api as sm

import statsmodels.formula.api as smf

model = sm.OLS(y, X)

results = model.fit()

print(results.summary())
X.loc[:,["MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd"]]

model = sm.OLS(y, X)

results = model.fit()

print(results.summary())