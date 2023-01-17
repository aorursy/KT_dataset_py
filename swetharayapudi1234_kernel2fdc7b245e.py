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
dataframe_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
dataframe_train.columns
dataframe_train['SalePrice'].describe()
sns.distplot(dataframe_train['SalePrice']);
print("Skewness: %f" % dataframe_train['SalePrice'].skew())

print("Kurtosis: %f" % dataframe_train['SalePrice'].kurt())
var = 'GrLivArea'

data = pd.concat([dataframe_train['SalePrice'], dataframe_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'TotalBsmtSF'

data = pd.concat([dataframe_train['SalePrice'], dataframe_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'OverallQual'

data = pd.concat([dataframe_train['SalePrice'], dataframe_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([dataframe_train['SalePrice'], dataframe_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
corrmat = dataframe_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(dataframe_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(dataframe_train[cols], size = 2.5)

plt.show();
total = dataframe_train.isnull().sum().sort_values(ascending=False)

percent = (dataframe_train.isnull().sum()/dataframe_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
dataframe_train = dataframe_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

dataframe_train = dataframe_train.drop(dataframe_train.loc[dataframe_train['Electrical'].isnull()].index)

dataframe_train.isnull().sum().max()
saleprice_scaled = StandardScaler().fit_transform(dataframe_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
var = 'GrLivArea'

data = pd.concat([dataframe_train['SalePrice'], dataframe_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
dataframe_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

dataframe_train = dataframe_train.drop(dataframe_train[dataframe_train['Id'] == 1299].index)

dataframe_train = dataframe_train.drop(dataframe_train[dataframe_train['Id'] == 524].index)
var = 'TotalBsmtSF'

data = pd.concat([dataframe_train['SalePrice'], dataframe_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
sns.distplot(dataframe_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(dataframe_train['SalePrice'], plot=plt)
dataframe_train['SalePrice'] = np.log(dataframe_train['SalePrice'])
sns.distplot(dataframe_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(dataframe_train['SalePrice'], plot=plt)
sns.distplot(dataframe_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(dataframe_train['GrLivArea'], plot=plt)
dataframe_train['GrLivArea'] = np.log(dataframe_train['GrLivArea'])
sns.distplot(dataframe_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(dataframe_train['GrLivArea'], plot=plt)
sns.distplot(dataframe_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(dataframe_train['TotalBsmtSF'], plot=plt)
dataframe_train['HasBsmt'] = pd.Series(len(dataframe_train['TotalBsmtSF']), index=dataframe_train.index)

dataframe_train['HasBsmt'] = 0 

dataframe_train.loc[dataframe_train['TotalBsmtSF']>0,'HasBsmt'] = 1
dataframe_train.loc[dataframe_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(dataframe_train['TotalBsmtSF'])
sns.distplot(dataframe_train[dataframe_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(dataframe_train[dataframe_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
plt.scatter(dataframe_train['GrLivArea'], dataframe_train['SalePrice']);
plt.scatter(dataframe_train[dataframe_train['TotalBsmtSF']>0]['TotalBsmtSF'], dataframe_train[dataframe_train['TotalBsmtSF']>0]['SalePrice']);
dataframe_train = pd.get_dummies(dataframe_train)