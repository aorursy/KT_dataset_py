# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

from matplotlib.pyplot import xticks

from scipy.stats import skew

import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import data



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
len_train = len(train)
train.head()
train.columns
sns.set(rc={'figure.figsize':(9,7)})

sns.distplot(train.SalePrice)
train.SalePrice = np.log(train.SalePrice)
sns.distplot(train.SalePrice)
#    A    Agriculture

#    C    Commercial

#    FV   Floating Village Residential

#    I    Industrial

#    RH   Residential High Density

#    RL   Residential Low Density

#    RP   Residential Low Density Park 

#    RM   Residential Medium Density



sns.countplot(train.MSZoning)
sns.catplot(x="MSZoning", y="SalePrice", kind="bar", data=train)
sns.set(rc={'figure.figsize':(25,8)})

sns.countplot(train.YearBuilt)

xticks(rotation=90)
sns.set(rc={'figure.figsize':(9,7)})

sns.countplot(train.TotRmsAbvGrd)
sns.catplot(x="TotRmsAbvGrd", y="SalePrice", kind="bar", data=train)
sns.set(rc={'figure.figsize':(9,7)})

sns.countplot(train.OverallQual)
sns.catplot(x="OverallQual", y="SalePrice", kind="bar", data=train)
# Regroup train and test in data

data = train.append(test)
data.isnull().sum().sort_values(ascending=False) 
round(data.isnull().sum().sort_values(ascending=False) / len(data),4)
data_na = data.isnull().sum() / len(data)

data_na = data_na.drop(data_na[data_na == 0].index)

data_na.sort_values(ascending=False)
data.PoolQC = data.PoolQC.fillna("None")
data.MiscFeature = data.MiscFeature.fillna("None")
data.Alley = data.Alley.fillna("None")
data.Fence = data.Fence.fillna("None")
data.FireplaceQu = data.FireplaceQu.fillna("None")
data.LotFrontage.describe()
data.groupby('Neighborhood')['LotFrontage'].mean().sort_values(ascending=False)
# We will replace missing values by the mean of LotFrontage of neighborhood properties



data.LotFrontage = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.mean()))
# In the data description, we have "NA   No Garage" so we replace the missing values by "None"
data.GarageFinish = data.GarageFinish.fillna('None')
# Same method for these columns



cols = ['GarageFinish', 'GarageQual', 'GarageType', 'GarageCond']



for col in cols:

    data[col] = data[col].fillna('None')
# Missing values mean no Garage



cols = ['GarageYrBlt', 'GarageCars', 'GarageArea']

for col in cols:

    data[col] = data[col].fillna(0)
# Missing values mean no Basement



cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

for col in cols:

    data[col] = data[col].fillna(0)
# Missing values mean no Basement



cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for col in cols:

    data[col] = data[col].fillna('None') 
data_na = data.isnull().sum() / len(data)

data_na = data_na.drop(data_na[data_na == 0].index)

data_na.sort_values(ascending=False)
data['MasVnrType'] = data['MasVnrType'].fillna('None')

data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
data.MSZoning.isnull().sum()
data.MSZoning.value_counts()
# We replace these values by the most frequent value in MSZoning



data.MSZoning = data.MSZoning.fillna(data.MSZoning.mode()[0])
data.Utilities.isnull().sum()
data.Utilities.value_counts()
print(data[data.Utilities == 'NoSeWa'])

print(train.shape)
data = data.drop('Utilities', axis=1)
data.Functional.isnull().sum()
data.Functional.value_counts()
# We will replace by the most common



data.Functional = data.Functional.fillna(data.Functional.mode()[0])
data.SaleType.isnull().sum()
data.SaleType.value_counts()
data.SaleType = data.SaleType.fillna(data.SaleType.mode()[0])
data.KitchenQual.isnull().sum()
data.KitchenQual.value_counts()
data.KitchenQual = data.KitchenQual.fillna(data.KitchenQual.mode()[0])
data.Electrical.isnull().sum()
data.Electrical.value_counts()
data.Electrical = data.Electrical.fillna(data.Electrical.mode()[0])
print(data.Exterior1st.isnull().sum())

print(data.Exterior2nd.isnull().sum())
print(data.Exterior1st.value_counts())

print(data.Exterior2nd.value_counts())
data.Exterior1st = data.Exterior1st.fillna(data.Exterior1st.mode()[0])

data.Exterior2nd = data.Exterior2nd.fillna(data.Exterior2nd.mode()[0])
data_na = data.isnull().sum() / len(data)

data_na = data_na.drop(data_na[data_na == 0].index)

data_na.sort_values(ascending=False)
data.dtypes
data.columns
data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF'] + data['GarageArea']
data = pd.get_dummies(data)
data.shape
train = data[:len_train]

test = data[len_train:]
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression, Ridge

import xgboost as xgb

import lightgbm as lgb
X_train = train.drop(['SalePrice'], axis=1)

y_train = train['SalePrice']

X_test = test.drop(['SalePrice'], axis=1)
# Linear Regression



linreg = LinearRegression()

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)

acc_lin = round(linreg.score(X_train, y_train) * 100, 2)

acc_lin
# Ridge



ridge = Ridge()

ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)

acc_ridge = round(ridge.score(X_train, y_train) * 100, 2)

acc_ridge
# Random Forest Regressor



random_forest  = RandomForestRegressor()

random_forest.fit(X_train, y_train)

y_pred = random_forest .predict(X_test)

acc_random_forest  = round(random_forest .score(X_train, y_train) * 100, 2)

acc_random_forest 
# Gradient Boosting Regressor



gbr  = GradientBoostingRegressor()

gbr.fit(X_train, y_train)

y_pred = gbr .predict(X_test)

acc_gbr  = round(gbr .score(X_train, y_train) * 100, 2)

acc_gbr 
# KNN

from sklearn.model_selection import cross_val_score



neighbors = range(1,20)

cv_scores = []



for k in neighbors:

    knn = KNeighborsRegressor(n_neighbors=k)

    scores = cross_val_score(knn, X_train, y_train)

    cv_scores.append(scores.mean())



# changing to misclassification error

MSE = [1 - x for x in cv_scores]



# determining best k

optimal_k = neighbors[MSE.index(min(MSE))]



print(optimal_k)
# plot misclassification error vs k

plt.plot(neighbors, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()
# We can choose 8 which is in the elbow of the curve, not different from 16
knn = KNeighborsRegressor(n_neighbors = 8)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn
# Random Forest Regressor



random_forest  = RandomForestRegressor()

random_forest.fit(X_train, y_train)

y_pred = np.exp(random_forest.predict(X_test))
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": y_pred

    })

submission.to_csv('submission.csv', index=False)