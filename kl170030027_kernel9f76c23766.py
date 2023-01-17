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
#bring in the six packs

a_train= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
#check the decoration

a_train.columns
#descriptive statistics summary

a_train['SalePrice'].describe()
#histogram

sns.distplot(a_train['SalePrice']);
#skewness and kurtosis

print("Skewness: %f" % a_train['SalePrice'].skew())

print("Kurtosis: %f" % a_train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([a_train['SalePrice'], a_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([a_train['SalePrice'], a_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([a_train['SalePrice'], a_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([a_train['SalePrice'], a_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
#correlation matrix

corrmat = a_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(a_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(a_train[cols], size = 2.5)

plt.show();
#missing data

total = a_train.isnull().sum().sort_values(ascending=False)

percent = (a_train.isnull().sum()/a_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#dealing with missing data

a_train = a_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

a_train = a_train.drop(a_train.loc[a_train['Electrical'].isnull()].index)

a_train.isnull().sum().max() #just checking that there's no missing data missing...
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(a_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([a_train['SalePrice'], a_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

a_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

a_train = a_train.drop(a_train[a_train['Id'] == 1299].index)

a_train = a_train.drop(a_train[a_train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([a_train['SalePrice'], a_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#histogram and normal probability plot

sns.distplot(a_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(a_train['SalePrice'], plot=plt)
#applying log transformation

a_train['SalePrice'] = np.log(a_train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(a_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(a_train['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(a_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(a_train['GrLivArea'], plot=plt)


#data transformation

a_train['GrLivArea'] = np.log(a_train['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(a_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(a_train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(a_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(a_train['TotalBsmtSF'], plot=plt)

#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

a_train['HasBsmt'] = pd.Series(len(a_train['TotalBsmtSF']), index=a_train.index)

a_train['HasBsmt'] = 0 

a_train.loc[a_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

a_train.loc[a_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(a_train['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(a_train[a_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(a_train[a_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#scatter plot

plt.scatter(a_train['GrLivArea'], a_train['SalePrice']);

#scatter plot

plt.scatter(a_train[a_train['TotalBsmtSF']>0]['TotalBsmtSF'], a_train[a_train['TotalBsmtSF']>0]['SalePrice']);
#convert categorical variable into dummy

a_train = pd.get_dummies(a_train)