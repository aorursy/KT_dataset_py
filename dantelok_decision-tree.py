# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv(("../input/house-prices-advanced-regression-techniques/train.csv"))

df_train = pd.DataFrame(train)

df_train
cols = df_train.columns

cols
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
# Personal judgement of choosing features

# Relationship between SalePrice(y) and numeric features

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x=var, y='SalePrice', ylim = (0, 800000))
#box plot with cat features

var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(40, 12))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x=var, y='SalePrice', ylim = (0, 800000))
var = 'GarageArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x=var, y='SalePrice', ylim = (0, 800000))
# Choose features with correlation matrix

corrmat = df_train.corr() #correlation matrix

f, ax = plt.subplots(figsize=(12, 9)) #graph size

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
total = df_train.isnull().sum().sort_values(ascending = False)

percentage = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percentage], axis = 1, keys = ['Total', 'Percentage'])

missing_data = missing_data[missing_data['Total'] > 0]

missing_data
# Delete data with >15% missing value, useless data and NaN rows

df_train = df_train.drop(missing_data[missing_data['Total'] > 1].index, 1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train
# check null data

df_train.isnull().sum().max()
# Option 1: Drop all NaN rows

df_train = df_train.dropna(axis = 1)

df_train
# Option 2: fill the NaN with 999/0

df_train.fillna(999)
# Option 3: drop the columns with NaN value

cols_with_missing = [col for col in df_train.columns

                                 if df_train[col].isnull().any()]

reduced_train_data = df_train.drop(cols_with_missing, axis=1)

reduced_train_data
# Outliers - Delete or keep it?

# Univariate Analysis - Standardize data ~ (0,1)

from sklearn.preprocessing import StandardScaler

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])   # outpput shape = (1459, 1)

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]                      # cannot argsort a [[x], [x], [x], ...] array => make it [x, x, x, ...]

high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('low range of distn')

print(low_range)

print('high range of distn')

print(high_range)
# Bivariate Analysis

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# Delete outlier-points

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
# Explore data

# Normality

# Homoscedasticity

# Linearity

# Absence of correlated errors
from scipy import stats

sns.distplot(df_train['SalePrice'], fit = stats.norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
# log transfromation (for +ve skewness)

df_train['SalePrice'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit = stats.norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
sns.distplot(df_train['GrLivArea'], fit = stats.norm)

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])



sns.distplot(df_train['GrLivArea'], fit = stats.norm)

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
sns.distplot(df_train['TotalBsmtSF'], fit = stats.norm)

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
# skewness

# zero values in observations (cannot do log trans)

# zero = no basement floor

# Create new variable classifying Having basement/No basemant

# df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

df_train['HasBsmt'] = 0

df_train.loc[df_train['TotalBsmtSF']>0, 'HasBsmt'] = 1

df_train
# Log Transformation for 'HasBsmt' == 1

df_train.loc[df_train['HasBsmt'] == 1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])



sns.distplot(df_train['GrLivArea'], fit = stats.norm)

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
# scatter plot for these combinations

plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice'])
# convert categorical variable into dummy variable

df_train = pd.get_dummies(df_train)

df_train
# Try ensembling DT and Regression model