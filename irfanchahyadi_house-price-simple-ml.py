import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import warnings

warnings.filterwarnings("ignore")
# Load dataset

df_train = pd.read_csv('../input/train.csv')

df_train['dataset'] = 'train'

df_test = pd.read_csv('../input/test.csv')

df_test['dataset'] = 'test'
df_train.shape
# Response (SalePrice)

distplot = sns.distplot(df_train.SalePrice, fit=stats.norm)
probplot = stats.probplot(df_train.SalePrice, plot=plt)
# get standard deviation

response = df_train[['SalePrice']].copy()

response['stddev'] = df_train[['SalePrice']].apply(lambda x: abs(x - x.mean())/x.std())

response[response.stddev > 3].sort_values('SalePrice', ascending=False).min()
# SalePrice >= 423000 is outlier

df_train = df_train[df_train.SalePrice < 423000]
# Transform response with log function

df_train['SalePrice'] = np.log(df_train['SalePrice'])
distplot = sns.distplot(df_train.SalePrice, fit=stats.norm)
# select all numeric features, including response

num_features= df_train.select_dtypes(['float64', 'int64']).keys()

len(num_features)
# select correlation > 0.5

num_corr = abs(df_train[num_features].corr()['SalePrice']).sort_values(ascending=False)

num_ok = num_corr[num_corr > 0.5].drop('SalePrice')

num_ok
for i in range(0, len(num_ok), 5):

    sns.pairplot(data=df_train, x_vars=num_ok[i:i+5].keys(), y_vars='SalePrice', kind='reg')
corr_plot = sns.heatmap(df_train[num_ok.keys()].corr(), cmap=plt.cm.Reds, annot=True)
# check NaN values on numerical falues

df_train[num_ok.keys()].isnull().sum()

train_num_null = df_train[num_ok.keys()].isnull().sum().sort_values(ascending=False)

test_num_null = df_test[num_ok.keys()].isnull().sum().sort_values(ascending=False)

print('null from train dataset:\n{}\n'.format(train_num_null[train_num_null > 0]))

print('null from test dataset:\n{}'.format(test_num_null[test_num_null > 0]))
# fill GarageYrBlt on train dataset

df_train['GarageYrBlt'] = df_train[['GarageYrBlt']].applymap(lambda x: 0 if pd.isnull(x) else x)

# fill GarageYrBlt, TotalBsmtSF, GarageArea, GarageCars on test dataset

for i in test_num_null[test_num_null > 0].keys():

    df_test[i] = df_test[[i]].applymap(lambda x: 0 if pd.isnull(x) else x)
cat_features = df_train.select_dtypes(['object'])

len(cat_features.keys())
# after see all plots, i will choose this features

cat_ok = ['MSZoning', 'LotShape', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond','HeatingQC', 'CentralAir', 'KitchenQual', 'GarageType', 'GarageFinish', 'SaleType', 'SaleCondition']
# this is how its look like

plt.rcParams['figure.max_open_warning'] = len(cat_ok)

for i in cat_ok:

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 3))

    sns.boxplot(x='SalePrice', y=i, data=df_train, ax=ax[0])

    sns.countplot(y=i, data=df_train, ax=ax[1])
# check NaN values on categorical features

train_cat_null = df_train[cat_ok].isnull().sum().sort_values(ascending=False)

test_cat_null = df_test[cat_ok].isnull().sum().sort_values(ascending=False)

print('null from train dataset:\n{}\n'.format(train_cat_null[train_cat_null > 0]))

print('null from test dataset:\n{}'.format(test_cat_null[test_cat_null > 0]))
print('GarageFinish:', df_train['GarageFinish'].unique())

print('GarageType:', df_train['GarageType'].unique())

print('BsmtCond:', df_train['BsmtCond'].unique())

print('BsmtQual:', df_train['BsmtQual'].unique())

print('MasVnrType:', df_train['MasVnrType'].unique())
# Replace NaN value to 'None'

for i in ['GarageFinish', 'GarageType', 'BsmtCond', 'BsmtQual', 'MasVnrType']:

    df_train[i] = df_train[[i]].applymap(lambda x: 'None' if pd.isnull(x) else x)

    df_test[i] = df_test[[i]].applymap(lambda x: 'None' if pd.isnull(x) else x)
print('MSZoning:', df_train.groupby(['MSZoning']).size())

print('\nKitchenQual:', df_train.groupby(['KitchenQual']).size())

print('\nSaleType:', df_train.groupby(['SaleType']).size())
df_test['MSZoning'] = df_test['MSZoning'].fillna(df_train['MSZoning'].mode()[0])

df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_train['KitchenQual'].mode()[0])

df_test['SaleType'] = df_test['SaleType'].fillna(df_train['SaleType'].mode()[0])
# concat train and test dataset

df_test['SalePrice'] = np.nan

df = pd.concat([df_train, df_test], sort=True)
# mean and stddev from numeric train dataset

num_mean = df_train[num_ok.index].mean()

num_std = df_train[num_ok.index].std()



# Standarize numeric features

df_num = (df[num_ok.keys()] - num_mean) / num_std
# Create dummies for categorical features

df_dummies = pd.get_dummies(df[cat_ok])
# dataset column + SalePrice + numeric features + categorical features

df = pd.concat([df.dataset, df.SalePrice, df_num, df_dummies], axis=1)

df.head()
# for splitting train and validate dataset

from sklearn.model_selection import train_test_split



# machine learning regressor

from sklearn.linear_model import LinearRegression, Lasso, ElasticNetCV

from sklearn.ensemble import RandomForestRegressor



# for evaluate model using RootMeanSquaredError

from sklearn.metrics import mean_squared_error
X = df[df.dataset == 'train'].drop(['dataset', 'SalePrice'], axis=1)

y = df[df.dataset == 'train'][['SalePrice']]

X_test = df[df.dataset == 'test'].drop(['dataset', 'SalePrice'], axis=1)
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

X_train, y_train = X, y
model = ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000)

model.fit(X_train, y_train)

# y_pred = model.predict(X_valid)

# mean_squared_error(y_valid, y_pred)
X_test.isnull().sum().sum()
# Create submission file

y_test_pred = np.exp(model.predict(X_test))

result = df_test[['Id']].copy()

result['SalePrice'] = y_test_pred

result.set_index('Id').to_csv('submission.csv')