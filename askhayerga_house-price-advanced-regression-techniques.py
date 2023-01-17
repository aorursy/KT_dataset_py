import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from scipy.stats import norm

import warnings

#ignore warnings

warnings.filterwarnings('ignore') 
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
# Skewness has to be equal to 0, Kurtosis has to be equal to 3 for normal distribution

print('Skewness: %f' % df_train['SalePrice'].skew())

print('Kurtosis: %f' % df_train['SalePrice'].kurt())
plt.subplots(figsize=(10, 5))

sns.distplot(df_train['SalePrice'], fit = norm)
plt.subplots(figsize=(10, 5))

prob = stats.probplot(df_train['SalePrice'], plot=plt)
df_train['SalePrice'] = np.log(df_train['SalePrice'])
plt.figure(1)

plt.subplots(figsize=(10, 5))

sns.distplot(df_train['SalePrice'], fit = norm)

plt.title('Normal')

plt.figure(2)

plt.subplots(figsize=(10, 5))

prob = stats.probplot(df_train['SalePrice'], plot=plt)
corrmat = df_train.corr()

plt.subplots(figsize=(10, 8))



k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)

plt.subplots(figsize=(8, 6))

plt.scatter(x='GrLivArea', y="SalePrice", data=data)

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')
data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)

plt.subplots(figsize=(8, 6))

plt.scatter(x='TotalBsmtSF', y="SalePrice", data=data)

plt.ylabel('SalePrice')

plt.xlabel('TotalBsmtSF')
data = pd.concat([df_train['SalePrice'], df_train['GarageArea']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

plt.scatter(x='GarageArea', y="SalePrice", data=data)

plt.ylabel('SalePrice')

plt.xlabel('GarageArea')
data = pd.concat([df_train['SalePrice'], df_train['GarageCars']], axis=1)

plt.subplots(figsize=(8, 6))

sns.boxplot(x='GarageCars', y="SalePrice", data=data)
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)

plt.subplots(figsize=(8, 6))

sns.boxplot(x='OverallQual', y="SalePrice", data=data)
df_all = pd.concat([df_train, df_test])

df_all = df_all.drop(columns= ['SalePrice', 'Id'])

df_all.shape
total = df_all.isnull().sum().sort_values(ascending=False)

missing_data = pd.DataFrame({'Total': total})

missing_data = missing_data[missing_data['Total'] > 0]

missing_data

df_all = pd.get_dummies(df_all)
df_all = df_all.fillna(df_all.mean())

# and check

df_all.isnull().sum().max()
X_train = df_all[:df_train.shape[0]]

X_test = df_all[df_train.shape[0]:]

y_train = df_train[['SalePrice']]
from sklearn.linear_model import Ridge

params = {'max_iter': 50000}

ridge = Ridge(**params)

est = GridSearchCV(ridge, param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]})

# number of params:

num = 100

print("Chosen parameter on %d datapoints: %s" % (num,est.fit(X_train[:num], y_train[:num]).best_params_))
params = {'alpha': 10.0, 'max_iter': 50000}

ridge = Ridge(**params)

# k-fold cross validation

kFold = 10 # number of sections
kf = KFold(n_splits=kFold, shuffle=True, random_state=2)

y_pr = np.zeros((X_train.shape[0],))
for train_index, test_index in kf.split(X_train):

    X_tr, X_te = X_train.iloc[train_index], X_train.iloc[test_index]

    y_tr, y_te = y_train.iloc[train_index], y_train.iloc[test_index]

    ridge.fit(X_tr, y_tr)

    #print(np.reshape(ridge.predict(X_te),-1))

    y_pr[test_index] = np.reshape(ridge.predict(X_te),-1) # in order to provide the same shape

err = sqrt(mean_squared_error(y_pr, y_train))

print("RMSE: %.5f" % err)
y_pr = ridge.predict(X_test)

result = np.exp(y_pr)
result = np.reshape(result, -1) # in order to provide the same shape in submission creating
result
submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': result})

submission.to_csv("submission_ridge.csv", index=None)