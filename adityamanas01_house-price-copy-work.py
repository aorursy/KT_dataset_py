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
train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.columns
train['SalePrice'].describe()
sns.distplot(train['SalePrice']);

sns.distplot(train['YearBuilt']);
print("skewness: %f" % train['SalePrice'].skew())

print("kurtosis: %f" % train['SalePrice'].kurt())
var ='GrLivArea'

data=pd.concat([train['SalePrice'],train[var]],axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0.800000));


var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
corrmat=train.corr()

f, ax=plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat,vmax=0.8, square=True)
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 3.5)

plt.show();
total=train.isnull().sum().sort_values(ascending=False)

percent=(train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
train=train.drop((missing_data[missing_data['Total'] > 1]).index,1)

train =train.drop(train.loc[train['Electrical'].isnull()].index)

train.isnull().max().sum()
train.isnull().max().sum()
#univariate analysis

saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([train['SalePrice'],train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#histogram and normal probability plot

sns.distplot(train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice']=np.log(train['SalePrice'])
sns.distplot(train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot=plt)
train['GrLivArea']=np.log(train['GrLivArea'])
sns.distplot(train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot=plt)
#if area>0 it gets 1, for area==0 it gets 0 

train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)

train['HasBsmt'] = 0 

train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#scatter plot

plt.scatter(train['GrLivArea'],train['SalePrice']);
#scatter plot

plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], train[train['TotalBsmtSF']>0]['SalePrice']);
train = pd.get_dummies(train)