# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_train.describe()
df_train.info()
labels = df_train.columns
for label in labels:
    print(label, "\t", df_train[label].isnull().sum())
from pandas.plotting import scatter_matrix
attribs = ['1stFlrSF', '2ndFlrSF', 'SalePrice', 'LotFrontage']
scatter_matrix(df_train[attribs], figsize = (16, 16))
corr_matrix = df_train.corr()
corr_matrix['SalePrice'].sort_values(ascending = False)
df_train = df_train.drop(['PoolArea', 'MoSold', '3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'Id', 'LowQualFinSF', 'YrSold', 'OverallCond', 'MSSubClass'], axis=1)
df_train.hist(bins = 50, figsize = (30,30))
plt.show()
plt.figure(figsize = (16, 8))
sns.scatterplot(x = df_train['YearBuilt'], y = df_train['SalePrice'], hue = df_train['1stFlrSF'])
plt.show()
#From the heatmap it looks like 
corr_matrix = df_train.corr()
plt.figure(figsize=(35, 20))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
current_palette = sns.color_palette("colorblind")
sns.heatmap(corr_matrix, annot=True)
#Check skewness and Kurtosis of price. Long tail!
print("Sale Price Skew: \t", df_train['SalePrice'].skew())
print("Sale Price Kurtosis: \t", df_train['SalePrice'].kurtosis())
#Check outliers in a histogram
plt.figure(figsize=(20,10))
sns.distplot(df_train['SalePrice'], bins=50, kde=False)
#Can we drop price above a certain value to eliminate outliers?
df_train.loc[(df_train['SalePrice']>500000), :]
#It doesn't seem like removing homes above 500k reduces the skewness
df_train.loc[(df_train['SalePrice']<500000), :].skew()
#Fill NaN with relevant values where Nan has meaning
df_train['Alley'].fillna("No Alley", inplace=True)
df_train['BsmtCond'].value_counts()
mask_bsmt = (df_train['BsmtExposure'].isnull())
cols_bsmt = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
             'BsmtUnfSF', 'TotalBsmtSF']
df_train.loc[mask_bsmt, cols_bsmt]
mask1_bsmt = (df_train['BsmtExposure'].isnull()) & (df_train['BsmtCond'].notnull())
df_train.loc[mask1_bsmt, 'BsmtExpsoure'] = 'No'
df_train.loc[mask_bsmt, cols_bsmt]
df_train['BsmtExposure'].value_counts()
for column in cols_bsmt:
    df_train[column].fillna('Nobasement')
cols = df_train.columns
for column in cols:
    if df_train[column].dtype == 'object':
        df_train[column] = df_train[column].astype(str)
df_train['MSZoning'] = df_train['MSZoning'].astype(str)
df_train["MSZoning"].dtype
