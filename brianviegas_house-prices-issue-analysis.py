import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from scipy.stats import norm
from scipy.stats import skew
pd.options.mode.chained_assignment = None  # default='warn'

file_path = '../input/train.csv' 
train_data = pd.read_csv(file_path)

correlation = train_data.corr()
correlation.sort_values(["SalePrice"], ascending = False, inplace = True)
print(correlation.SalePrice)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(correlation, vmax=.8, square=True);
plt.show()
print("Missing values count of OverallQual = ", train_data.OverallQual.isnull().sum())
print("Missing values count of GrLivArea = ", train_data.GrLivArea.isnull().sum())

key = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[key]], axis=1)
data.plot.scatter(x=key, y='SalePrice', ylim=(0,800000))

key = 'OverallQual'
data = pd.concat([train_data['SalePrice'], train_data[key]], axis=1)
data.plot.scatter(x=key, y='SalePrice', ylim=(0,800000))
train_data = train_data[train_data.GrLivArea < 4000]
print("Skew = ", train_data['OverallQual'].skew())
sns.distplot(train_data['OverallQual'], fit=norm)
print("Skew = ", train_data['GrLivArea'].skew())
sns.distplot(train_data['GrLivArea'], fit=norm)
train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
print("Skew = ", train_data['GrLivArea'].skew())
sns.distplot(train_data['GrLivArea'], fit=norm)
print("Skew = ", train_data['SalePrice'].skew())
sns.distplot(train_data['SalePrice'], fit=norm)
train_data['SalePrice'] = np.log(train_data['SalePrice'])
print("Skew =  ", train_data['SalePrice'].skew())
sns.distplot(train_data['SalePrice'], fit=norm)
missing_values_count = train_data.isnull().sum().sort_values(ascending=False)
missing_values_percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing_values_count, missing_values_percent], axis=1, keys=['Total', 'Percent'])
print(missing_data[missing_data['Total'] > 0])
print(train_data.isnull().sum().max())
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'):
    train_data[col] = train_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    train_data[col] = train_data[col].fillna(0)
train_data["LotFrontage"] = train_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
print(train_data.isnull().sum().max())
numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
train_data_numerical = train_data[numerical_features]
skewness = train_data_numerical.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
skewed_features = skewness.index
train_data_numerical[skewed_features] = np.log1p(train_data_numerical[skewed_features])
categorical_features = train_data.select_dtypes(include = ["object"]).columns
train_data_categorical = train_data[categorical_features]
train_data_categorical = pd.get_dummies(train_data_categorical)

# Join categorical and numerical variables
train_data = pd.concat([train_data_numerical, train_data_categorical], axis = 1)
print("NAs count of features in data set : ", train_data.isnull().values.sum())