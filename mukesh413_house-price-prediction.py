# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
text = open("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt", 'r')

print(text.read())
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
print(df_train.columns)

print(len(df_train.columns))
df_train['SalePrice'].describe()
#histogram

sns.distplot(df_train['SalePrice'])
#skewness and kutosis measure

print(df_train['SalePrice'].skew())

print(df_train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000))
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000))
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

f, ax = plt.subplots(figsize = (8, 6))

fig = sns.boxplot(x = var, y = 'SalePrice', data = data)

fig.axis(ylim = (0, 800000))
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

f, ax = plt.subplots(figsize = (25, 8))

fig = sns.boxplot(x = var, y = 'SalePrice', data = data)

fig.axis(ylim = (0, 800000))
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize = (10, 10))

sns.heatmap(corrmat, vmax = 0.8, square = True)
#saleprice correlation matrix

k = 10 #no of variables in the heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale = 1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatter plots

sns.set()

cols = ['SalePrice', 'OverallQual','GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show()
#total missing values column-wise

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

missing_data.head(20)
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.shape
df_train.isnull().sum().max()
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])
StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)