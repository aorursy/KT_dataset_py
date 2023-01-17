import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques//train.csv')

df_train.columns
total = df_train.isnull().sum().sort_values(ascending=False) 

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False) 

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(15)
drop_columns = ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","LotFrontage"]

df_train = df_train.drop(drop_columns, axis=1)
category = "MSZoning"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean()  
category = "Street"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "LotShape"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "LandContour"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "Utilities"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "LotConfig"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "LandSlope"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "Neighborhood"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "Condition1"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "Condition2"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "BldgType"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "HouseStyle"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "RoofStyle"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "RoofMatl"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "Exterior1st"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "Exterior2nd"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "MasVnrType"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean()
category = "ExterQual"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "ExterCond"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "Foundation"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "BsmtQual"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "BsmtCond"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "BsmtCond"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "BsmtFinType1"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "BsmtFinType2"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "BsmtFinType2"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "HeatingQC"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "CentralAir"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "Electrical"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "KitchenQual"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "Functional"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "GarageType"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "GarageFinish"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "GarageQual"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "GarageCond"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "PavedDrive"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "SaleType"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
category = "SaleCondition"

print(df_train[category].value_counts())

df_train[[category,"SalePrice"]].groupby([category]).mean() 
corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()