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
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm

import matplotlib.animation as animation

from matplotlib import rc

import unittest

from pylab import rcParams

%matplotlib inline



from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
sample = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

train =pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train.head()
train.shape
# Analise e visualizacao dos dados
train.columns
train['SalePrice'].describe()
plt.figure(figsize=(20,5))

sns.distplot(train.SalePrice, color="green")

plt.title("Target distribution in train")

plt.ylabel("Density");
plt.figure(figsize=(20,5))

sns.distplot(np.log(train.SalePrice), color="green")

plt.title("Log Target distribution in train")

plt.ylabel("Density");
var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), s=32);
var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 5))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
# matrix de correlaçao

matrix_corr = train.corr()

f, ax = plt.subplots(figsize = (14,8))

sns.heatmap(matrix_corr, vmax = .8, square = True)
# matriz de correlacao do preco

k = 10 # numero de variaveis para o heatmap

col = matrix_corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[col].values.T)

sns.set(font_scale = 1)

hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size':10}, yticklabels = col.values, xticklabels = col.values)

plt.show()
sns.set()

col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[col], size = 2)

plt.show()
# maior correlacao

corr = train.corr()

print(corr['SalePrice'].sort_values(ascending = False)[:5])
#histogram and normal probability plot

sns.distplot(train['SalePrice'], fit = norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot = plt)
#applying log transformation

train['SalePrice'] = np.log(train['SalePrice'])



#transformed histogram and normal probability plot

sns.distplot(train['SalePrice'], fit = norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot = plt)
#scatter plot

plt.scatter(train['GrLivArea'], train['SalePrice']);