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
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
df_train.columns
df_train.SalePrice.describe()
# Histogram

sns.distplot(df_train.SalePrice);
#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
sns.scatterplot(df_train.SalePrice, df_train.GrLivArea);
sns.scatterplot(df_train.SalePrice, df_train.TotalBsmtSF);
sns.boxplot(df_train.OverallQual, df_train.SalePrice);
f, ax = plt.subplots(figsize = (16,8))

sns.boxplot(df_train.YearBuilt, df_train.SalePrice);
corrmat = df_train.corr()

f, ax = plt.subplots(figsize = (12, 9))

sns.heatmap(corrmat, vmax=.8);
k = 10 # number of vars in the heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#missing data

total = df_train[cols].isnull().sum().sort_values(ascending=False)

percent = (df_train[cols].isnull().sum()/df_train[cols].isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
# missing data on test set

if 'SalePrice' in cols:

  cols.remove('SalePrice')



total = df_test[cols].isnull().sum().sort_values(ascending=False)

percent = (df_test[cols].isnull().sum()/df_test[cols].isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
df_test.TotalBsmtSF = df_test.TotalBsmtSF.fillna(0)

df_test.GarageCars = df_test.GarageCars.fillna(0)
# missing data on test set

total = df_test[cols].isnull().sum().sort_values(ascending=False)

percent = (df_test[cols].isnull().sum()/df_test[cols].isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
from scipy import stats

from scipy.stats import norm



# histogram and normal probability plot

sns.distplot(df_train.SalePrice, fit=norm);

fig = plt.figure()

res = stats.probplot(df_train.SalePrice, plot=plt)
df_train.SalePrice = np.log(df_train.SalePrice)
# histogram and normal probability plot

sns.distplot(df_train.SalePrice, fit=norm);

fig = plt.figure()

res = stats.probplot(df_train.SalePrice, plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#if area>0 it gets 1, for area==0 it gets 0

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

df_train['HasBsmt'] = 0 

df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF']);
#histogram and normal probability plot

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#convert categorical variable into dummy

df_train = pd.get_dummies(df_train)
df_train.head()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



if 'SalePrice' in cols:

  cols.remove('SalePrice')

X = df_train[cols]

y = df_train.SalePrice



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=39)



X_test = df_test[cols]



# sc = StandardScaler()

# X_train = sc.fit_transform(X_train)

# X_val = sc.transform(X_val)

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR



regressor = RandomForestRegressor(n_estimators=200, random_state=0)

regressor.fit(X_train, y_train)

y_pred = np.exp(regressor.predict(X_val))

y_val = np.exp(y_val)
from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_pred)))

print('RMSE/Mean (Lower is Better):', np.sqrt(metrics.mean_squared_error(y_val, y_pred)) / y_val.mean() )
y_test_pred = np.exp(regressor.predict(X_test))
sub = pd.DataFrame()

sub['Id'] = df_test.Id

sub['SalePrice'] = y_test_pred
sub
sub.to_csv('submission.csv',index=False)