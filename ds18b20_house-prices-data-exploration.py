# Reference from: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_train.shape
# print all features
df_train.columns
# print all features's data types
df_train.dtypes
df_train['SalePrice'].describe()
# show histogram graph of SalePrice
sns.distplot(df_train['SalePrice'])
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
# scatter plot GrLivArea/SalePrice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# scatter plot TotalBsmtSF/SalePrice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# scatter plot GarageArea/SalePrice
var = 'GarageArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# scatter plot OverallQual/SalePrice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
# correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)  # need to learn
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot
# sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()
# missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)

# concatenate two series
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
#standardizing data
saleprice = df_train['SalePrice'].values
saleprice_scaled = saleprice - saleprice.mean()
print('saleprice_scaled[:10]:', saleprice_scaled[:10])
print('saleprice_scaled.std():', saleprice_scaled.std())
saleprice_scaled = saleprice_scaled / saleprice_scaled.std()
x = np.sort(saleprice_scaled)

idx = np.argsort(saleprice_scaled)
print('start 10:', saleprice_scaled[idx[:10]])
print('end 10:', saleprice_scaled[idx[-10:]])
#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train.shape
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train.shape
df_train = df_train.drop(df_train[df_train['Id'] == 523].index)
df_train.shape
#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# histogram plot
sns.distplot(df_train['SalePrice'], fit=norm);
# probability plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
# log 10
df_train['SalePrice'] = np.log(df_train['SalePrice'])
# histogram plot
sns.distplot(df_train['SalePrice'], fit=norm);
# probability plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
# histogram plot
sns.distplot(df_train['GrLivArea'], fit=norm);
# probability plot
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
# log 10
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
# histogram plot
sns.distplot(df_train['GrLivArea'], fit=norm);
# probability plot
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
#convert categorical variable into dummy
df_train = pd.get_dummies(df_train)
df_train.shape
