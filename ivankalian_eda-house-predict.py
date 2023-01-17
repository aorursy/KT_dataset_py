import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

from sklearn import preprocessing

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(df_train.shape, df_test.shape)
df_train.mean()
total = df_train.isnull().sum()

procent = df_train.isnull().sum() / df_train.notnull().sum()

missing_data = pd.concat([total, procent], axis=1, keys=['Total', 'Percent'])

# pd.concat([missing_data[missing_data['Percent'] > 0], missing_data[missing_data['Percent'] != 0.000685]], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Total'] > 0]
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000)); #линейная зависимость
sns.distplot(df_train['SalePrice'])
corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(13, 10))

sns.heatmap(corrmat, vmax=.8, square=True)
k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars']

sns.pairplot(df_train[cols], size = 2.5)

plt.show()