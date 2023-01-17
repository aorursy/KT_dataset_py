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
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.columns
train['SalePrice'].describe()
#histogram

sns.distplot(train['SalePrice'])
# skewness and kurtosis

print('Skewnees: %f' % train['SalePrice'].skew())

print('Kurtosis: %f' % train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice

# GriLivArea = 상층(지상) 거주 면적 평방 피트

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim=(0,800000))
# TotalBsmtSF = 지하 면적의 총 제곱피트

var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
#box plot overallqual/saleprice

# 전체 소재 및 마감 퀄리티(점수인가?) -> 집의 전반적인 퀄리티 점수인듯..

var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis = 1)

f, ax = plt.subplots(figsize = (8, 6))

fig = sns.boxplot(x = var, y = 'SalePrice', data = data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

# YearBuilt = 공사 일자

data = pd.concat([train['SalePrice'], train[var]], axis = 1)

f, ax = plt.subplots(figsize = (25,8))

fig = sns.boxplot(x = var, y = 'SalePrice', data = data)

fig.axis(ymin = 0 , ymax = 800000)

plt.xticks(rotation = 90)
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize = (12,9))

sns.heatmap(corrmat, vmax = .8, square = True)
# saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale = 1.25)

hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size':10}, yticklabels = cols.values, xticklabels = cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

# missing data

total = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

missing_data.head(20)
# dealing with missing data

train = train.drop((missing_data[missing_data['Total']>1]).index,1)

train = train.drop(train.loc[train['Electrical'].isnull()].index)

train.isnull().sum().max()
# standardizing data

# StandardScaler() -> 전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정.

# 스케일링은 자료의 overflow, underflow를 방지, 독립 변수의 공분산 행렬의 조건수를 감소시켜 최적화 과정에서의 안정성 및 수렴 속도 향상

saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis])

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)

sns.distplot(saleprice_scaled)
var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
#deleting points

train.sort_values(by = 'GrLivArea', ascending = False)[:2]

train = train.drop(train[train['Id'] == 1299].index)

train = train.drop(train[train['Id'] == 524].index)
#bivariate analysis saleprice/TotalBsmtSF

var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
#histogram and normal probability plot

sns.distplot(train['SalePrice'], fit = norm)

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot = plt)
#applying log transformation

train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'], fit = norm)

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot = plt)
#histogram and normal probability plot

sns.distplot(train['GrLivArea'], fit = norm)

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot = plt)
train['GrLivArea'] = np.log(train['GrLivArea'])
sns.distplot(train['GrLivArea'], fit = norm)

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot = plt)
#histogram and normal probability plot

sns.distplot(train['TotalBsmtSF'], fit = norm)

fig = plt.figure()

res = stats.probplot(train['TotalBsmtSF'], plot = plt)
#create column for new varibale (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area == 0 it gets 0

train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index = train.index)

train['HasBsmt'] = 0

train.loc[train['TotalBsmtSF']>0, 'HasBsmt'] = 1
#transform data

train.loc[train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#scatter plot

plt.scatter(train['GrLivArea'], train['SalePrice'])
#scatter plot

plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], train[train['TotalBsmtSF']>0]['SalePrice'])
#convert categorical variable into dummy

train = pd.get_dummies(train)
train