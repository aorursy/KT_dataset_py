# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

data.head()
data.info()

sns.distplot(data['SalePrice'])
data['SalePrice'].skew()
data['SalePrice'].kurt()
var = 'GrLivArea'

sns.scatterplot(data[var], data['SalePrice'], marker='o')
var = 'TotalBsmtSF'

sns.scatterplot(data[var], data['SalePrice'], marker='o')
sns.boxplot(data['OverallQual'], data['SalePrice'])
var = 'YearBuilt'

sns.scatterplot(data[var], data['SalePrice'])
corr_mat = data.corr()

plt.subplots(figsize=(12,12))

sns.heatmap(corr_mat, cmap='YlGnBu')
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corr_mat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(data[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': k}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
cols = corr_mat.nlargest(k, 'SalePrice')

cols
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(data[cols], height = 2.5)

plt.show();

df_null = data.isnull().sum()

df_null = df_null[df_null != 0].sort_values(ascending = False)

df_null = df_null/len(data)*100

df_null

# Percentage of missing/null values
cols_drop = list(df_null.index[:6].values)

for col in cols_drop:

    data = data.drop(col, axis=1)
#Electrical remove one missing value

ind = data.loc[data['Electrical'].isnull()].index

data = data.drop(ind)
cols_drop
# Fixing skewness

from scipy.stats import norm

from scipy import stats

data.head()

# sns.distplot(, fit=norm);

# fig = plt.figure()

# res = stats.probplot(data['SalePrice'], plot=plt)
#applying log transformation

data['SalePrice'] = np.log(data['SalePrice'])

sns.distplot(data['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(data['SalePrice'], plot=plt)
df = pd.get_dummies(data[cols[1:]])

df.head()
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(df, data['SalePrice'])

rf.score(df, data['SalePrice'])
data_test = pd.read_csv('../input/test.csv')

# df_test.isnull().sum().sort_values(ascending=False)
df_test = pd.get_dummies(df_test[cols[1:]])

df_test.head()
df_test.loc[df_test['GarageCars'].isnull(), 'GarageCars'] = 1.0

df_test.loc[df_test['TotalBsmtSF'].isnull(), 'TotalBsmtSF'] = 896
df_test.isnull().sum()
prices = rf.predict(df_test)

prices = np.exp(prices)

result = pd.DataFrame()

result['Id'] = data_test['Id']

result['SalePrice'] = prices
result.head()