# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from scipy.stats import norm

from scipy import stats
df_train = pd.read_csv('../input/train.csv')
df_train.columns
df_train.info
df_train.describe()
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter (x=var, y='SalePrice', ylim = (0,800000))
var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

#data.head(5)

data.plot.scatter(x=var, y='SalePrice', ylim = (0,800000))
var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

#data.head(5)

f, ax = plt.subplots(figsize = (20,6))

fig = sns.boxplot(x = var, y = 'SalePrice', data = data)

fig.axis(ymin=0, ymax=800000)
corrmat = df_train.corr()

f, ax = plt.subplots(figsize = (12,9))

sns.heatmap (corrmat, vmax = 0.8, square = True)
k = 10

cols = corrmat.nlargest (k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale = 1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 3)

plt.show();
total = df_train.isnull().sum().sort_values(ascending = False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

missing_data.head(20)
df_train = df_train.drop((missing_data[missing_data['Total'] >1]).index, 1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max()
#Univariable analysis

from sklearn.preprocessing import StandardScaler

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x=var, y = 'SalePrice', ylim = (0,800000))

plt.show()
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 424].index)
var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))

plt.show()
df_train.sort_values(by = 'TotalBsmtSF',ascending = False)[:4]
df_train = df_train.drop(df_train[df_train['TotalBsmtSF'] > 3000].index)

df_train.sort_values(by = 'TotalBsmtSF', ascending = False)[:4]
sns.distplot(df_train['SalePrice'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot = plt)
#'SalePrice' is not normal. It shows 'Peakedness', positive skewness and doesn't follow the diagonal line.

#in case of positive skewness, log transformations usually works well.

#applying log transformation

df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot = plt)
sns.distplot(df_train['GrLivArea'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot = plt)
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
sns.distplot (df_train['GrLivArea'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot = plt)
sns.distplot(df_train['TotalBsmtSF'], fit = norm)

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot = plt)
print ('skewness of TotalBsmtSF is %f' % df_train['TotalBsmtSF'].skew())

print ('skewness of SalePrice is %f' % df_train['SalePrice'].skew())

print('skewness of GrLivArea is %f' % df_train['GrLivArea'].skew())
#High risk engineering for HasBsmt

#log transformation to all non-zero observations, ignoring those with value zero

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index = df_train.index)

df_train['HasBsmt'] = 0

df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
sns.distplot(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit = norm)

fig = plt.figure()

res = stats.probplot (df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot = plt)
plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
plt.scatter(df_train[df_train['TotalBsmtSF'] > 0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF'] > 0]['SalePrice'])
df_train = pd.get_dummies(df_train)
from sklearn import preprocessing

from sklearn import linear_model, svm, gaussian_process

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

x = df_train[cols].values

y = df_train['SalePrice'].values

x_scaled = preprocessing.StandardScaler().fit_transform(x)

y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size = 0.3, random_state = 42)
clfs = {

        'svm': svm.SVR(),

        'RandomForestRegressor': RandomForestRegressor(n_estimators = 400),

        }

for clf in clfs:

    try:

        clfs[clf].fit(x_train, y_train)

        y_pred = clfs[clf].predict(x_test)

        print(clf + ' cost:' + str(np.sum(y_pred - y_test)/len(y_pred)))

    except Exception as e:

        print(clf + 'Error')

        pritn(str(e))
clf = RandomForestRegressor(n_estimators = 400)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_train)

print(y_pred)
y_test
f,ax = plt.subplots(figsize = (15,8))

fig = plt.scatter(y_train, y_pred)

plt.xlabel('y_train')

plt.xlabel('prediction')

plt.show()
rfr = clf
import pandas as pd

df_test = pd.read_csv('../input/test.csv')
df_test.info()
total_test = df_test[cols].isnull().sum().sort_values(ascending = False)

total_test.head(10)
df_test.head(20)
df_test = df_test[cols].fillna(df_test.mean())
df_test
df_test[cols].isnull().sum().max()
x = df_test.values

y_test_pred = rfr.predict(x)

print(y_test_pred)
print(y_test_pred.shape)
print(x.shape)
prediction = pd.DataFrame(y_test_pred, columns = ['SalePrice'])

result = pd.concat([df_test[:0], prediction], axis = 1)

result = result.drop(cols, axis = 1)

result.head(5)