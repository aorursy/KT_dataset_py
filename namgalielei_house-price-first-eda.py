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



import seaborn as sns

import matplotlib.pyplot as plt
sns.set()
def ecdf(data):

    x = np.sort(data)

    y = np.arange(0, len(x),1) / len(x)

    return x, y
def plot_histogram_train_test(feature, bins=100):

    # train and test histogram

    plt.figure(figsize=(20, 3))

    plt.subplot(1,2,1)

    plt.title('Train distribution')

    plt.xlabel(feature)

    plt.hist(train_df[feature], bins=bins)

    plt.subplot(1,2,2)

    plt.hist(test_df[feature], bins=bins)

    plt.title('Test distribution')

    plt.xlabel(feature)

    

def plot_kde_train_test(feature):

    # train and test kde

    plt.figure(figsize=(20, 3))

    plt.subplot(1,2,1)

    plt.title('Train distribution')

    plt.xlabel(feature)

    sns.kdeplot(train_df[feature])

    plt.subplot(1,2,2)

    sns.kdeplot(test_df[feature])

    plt.title('Test distribution')

    plt.xlabel(feature)

    

def plot_ecdf_train_test(feature):

    # train and test ecdf

    plt.figure(figsize=(20, 3))

    plt.subplot(1,2,1)

    plt.title('Train distribution')

    plt.xlabel(feature)

    plt.scatter(*ecdf(train_df[feature]))

    plt.subplot(1,2,2)

    plt.scatter(*ecdf(test_df[feature]))

    plt.title('Test distribution')

    plt.xlabel(feature)

    

def count_plot_train_test(feature):

    # train and test countplot

    plt.figure(figsize=(20, 3))

    plt.subplot(1,2,1)

    plt.title('Train distribution')

    plt.xlabel(feature)

    sns.countplot(train_df[feature])

    plt.subplot(1,2,2)

    sns.countplot(test_df[feature])

    plt.title('Test distribution')

    plt.xlabel(feature)

    

def checknull(feature):

    print('Portion of null values in train and test respectively:', 

          train_df[feature].isnull().mean(), ',', 

          test_df[feature].isnull().mean())
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_df.head()
test_df.head()
test_df.info()
print('Number of columns in dataset:', len(train_df.columns))

print('Including 1 target columns: SalePrice')
# drop columns having null value > 30%

null_thresh = 0.3



is_nan_train = train_df.isnull().mean()

dropped_cols_train = is_nan_train[is_nan_train > null_thresh].index

is_nan_test = test_df.isnull().mean()

dropped_cols_test = is_nan_test[is_nan_test > null_thresh].index



print('Columns having null value > '+str(null_thresh),'in train_df:',dropped_cols_train)

print('Columns having null value > '+str(null_thresh),'in test_df:',dropped_cols_test)

# print('\nPortion of null of these columns in test_df:\n', test_df[dropped_cols].isnull().mean())

# train_df.drop(dropped_cols, axis=1, inplace=True)
train_df.drop(dropped_cols_train, axis=1, inplace=True)

test_df.drop(dropped_cols_test, axis=1, inplace=True)
train_df.info()
train_corr_matrix = train_df.corr()

valuable_cols = train_corr_matrix[np.abs(train_corr_matrix['SalePrice']) > 0.3].index

print('Columns having correlation with SalePrice > 0.3: ', valuable_cols.values)

train_df = train_df[valuable_cols]

test_df = test_df[valuable_cols[:-1]]
train_corr_matrix[np.abs(train_corr_matrix['SalePrice']) > 0.55].index
sns.heatmap(train_df.corr())
train_df.head()
train_df.LotFrontage.dtype
print('Portion of null values in train and test respectively:', train_df.LotFrontage.isnull().mean(), ',', test_df.LotFrontage.isnull().mean())
# train and test LotFrontage distribution

plt.figure(figsize=(20, 3))

plt.subplot(1,2,1)

plt.title('Train distribution')

plt.xlabel('LotFrontage')

sns.kdeplot(train_df['LotFrontage'])

plt.subplot(1,2,2)

sns.kdeplot(test_df['LotFrontage'])

plt.title('Test distribution')

plt.xlabel('LotFrontage')
# train and test LotFrontage histogram

plt.figure(figsize=(20, 3))

plt.subplot(1,2,1)

plt.title('Train distribution')

plt.xlabel('LotFrontage')

plt.hist(train_df['LotFrontage'], bins=100)

plt.subplot(1,2,2)

plt.hist(test_df['LotFrontage'], bins=100)

plt.title('Test distribution')

plt.xlabel('LotFrontage')
# train and test LotFrontage histogram

plt.figure(figsize=(13, 3))



plt.subplot(1,2,1)

plt.scatter(*ecdf(train_df['LotFrontage']))

plt.title('Train cumulative plot')

plt.xlabel('LotFrontage')



plt.subplot(1,2,2)

plt.scatter(*ecdf(test_df['LotFrontage']))

plt.title('Test cumulative plot')

plt.xlabel('LotFrontage')
# fir a regression line

plt.figure(figsize=(10,6))

sns.scatterplot('LotFrontage', 'SalePrice', data=train_df)

sns.regplot('LotFrontage', 'SalePrice', data=train_df, scatter=None)
train_df.OverallQual.dtype
print('Portion of null values in train and test respectively:', train_df.OverallQual.isnull().mean(), ',', test_df.OverallQual.isnull().mean())
plt.figure(figsize=(10, 5))



plt.subplot(1,2,1)

sns.countplot(train_df.OverallQual)

plt.title('Value count of each OverallQual type on train set')



plt.subplot(1,2,2)

sns.countplot(test_df.OverallQual)

plt.title('Value count of each OverallQual type on train set')
train_df.groupby('OverallQual').mean()['SalePrice']
sns.scatterplot('OverallQual', 'SalePrice', data=train_df)
train_df.YearBuilt.dtype
print('Portion of null values in train and test respectively:', train_df.YearBuilt.isnull().mean(), ',', test_df.YearBuilt.isnull().mean())
np.sort(test_df.YearBuilt.unique())
np.sort(train_df.YearBuilt.unique())
# train and test histogram

plt.figure(figsize=(13, 3))



plt.subplot(1,2,1)

plt.scatter(*ecdf(train_df.YearBuilt))

plt.title('Train cumulative plot')

plt.xlabel('YearBuilt')



plt.subplot(1,2,2)

plt.scatter(*ecdf(test_df.YearBuilt))

plt.title('Test cumulative plot')

plt.xlabel('YearBuilt')
sns.scatterplot('YearBuilt', 'SalePrice', data=train_df)
train_df.YearRemodAdd.dtype
print('Portion of null values in train and test respectively:', train_df.YearRemodAdd.isnull().mean(), ',', test_df.YearRemodAdd.isnull().mean())
# train and test histogram

plt.figure(figsize=(13, 3))



plt.subplot(1,2,1)

plt.scatter(*ecdf(train_df.YearRemodAdd))

plt.title('Train cumulative plot')

plt.xlabel('YearRemodAdd')



plt.subplot(1,2,2)

plt.scatter(*ecdf(test_df.YearRemodAdd))

plt.title('Test cumulative plot')

plt.xlabel('YearRemodAdd')
sns.scatterplot('YearRemodAdd', 'SalePrice', data=train_df)
plt.scatter(train_df['YearRemodAdd']-train_df['YearBuilt'], train_df['SalePrice'])
train_df.MasVnrArea.dtype
print('Portion of null values in train and test respectively:', train_df.MasVnrArea.isnull().mean(), ',', test_df.MasVnrArea.isnull().mean())
# train and test histogram

plt.figure(figsize=(13, 3))



plt.subplot(1,2,1)

plt.scatter(*ecdf(train_df.MasVnrArea))

plt.title('Train cumulative plot')

plt.xlabel('MasVnrArea')



plt.subplot(1,2,2)

plt.scatter(*ecdf(test_df.MasVnrArea))

plt.title('Test cumulative plot')

plt.xlabel('MasVnrArea')
# train and test MasVnrArea histogram

plt.figure(figsize=(20, 3))

plt.subplot(1,2,1)

plt.title('Train distribution')

plt.xlabel('MasVnrArea')

plt.hist(train_df['MasVnrArea'], bins=100)

plt.subplot(1,2,2)

plt.hist(test_df['MasVnrArea'], bins=100)

plt.title('Test distribution')

plt.xlabel('MasVnrArea')
(train_df.MasVnrArea == 0).sum()
(test_df.MasVnrArea == 0).sum()
train_df.BsmtFinSF1.dtype
checknull('BsmtFinSF1')
plot_kde_train_test('BsmtFinSF1')
plot_histogram_train_test('BsmtFinSF1')
plot_ecdf_train_test('BsmtFinSF1')
sns.lmplot('BsmtFinSF1', 'SalePrice', data=train_df)
train_df.TotalBsmtSF.dtype
checknull('TotalBsmtSF')
plot_histogram_train_test('TotalBsmtSF')
plot_kde_train_test('TotalBsmtSF')
plot_ecdf_train_test('TotalBsmtSF')
sns.lmplot('TotalBsmtSF', 'SalePrice', train_df)
train_df['1stFlrSF'].dtype
checknull('1stFlrSF')
plot_histogram_train_test('1stFlrSF')
plot_kde_train_test('1stFlrSF')
plot_ecdf_train_test('1stFlrSF')
sns.lmplot('1stFlrSF', 'SalePrice', train_df)
train_df['2ndFlrSF'].dtype
checknull('2ndFlrSF')
plot_histogram_train_test('2ndFlrSF')
plot_kde_train_test('2ndFlrSF')
plot_ecdf_train_test('2ndFlrSF')
sns.lmplot('2ndFlrSF', 'SalePrice', train_df)
train_df.GrLivArea.dtype
checknull('GrLivArea')
plot_histogram_train_test('GrLivArea')
plot_kde_train_test('GrLivArea')
plot_ecdf_train_test('GrLivArea')
sns.lmplot('GrLivArea', 'SalePrice', train_df)
train_df.FullBath.dtype
checknull('FullBath')
count_plot_train_test('FullBath')
sns.lmplot('FullBath', 'SalePrice', train_df)
train_df.TotRmsAbvGrd.dtype
train_df.TotRmsAbvGrd.unique()
checknull('TotRmsAbvGrd')
count_plot_train_test('TotRmsAbvGrd')
sns.lmplot('TotRmsAbvGrd', 'SalePrice', data=train_df)
train_df.Fireplaces.dtype
train_df.Fireplaces.unique()
checknull('Fireplaces')
count_plot_train_test('Fireplaces')
sns.lmplot('Fireplaces', 'SalePrice', train_df)
train_df.GarageYrBlt.dtype
train_df.GarageYrBlt.unique()
plot_histogram_train_test('GarageYrBlt')
plot_kde_train_test('GarageYrBlt')
plot_ecdf_train_test('GarageYrBlt')
sns.lmplot('GarageYrBlt', 'SalePrice', train_df)
train_df.GarageCars.dtype
train_df.GarageCars.unique()
checknull('GarageCars')
count_plot_train_test('GarageCars')
sns.lmplot('GarageCars', 'SalePrice', train_df)
train_df.GarageArea.dtype
checknull('GarageArea')
plot_histogram_train_test('GarageArea')
plot_kde_train_test('GarageArea')
plot_ecdf_train_test('GarageArea')
sns.lmplot('GarageArea', 'SalePrice', train_df)
train_df.WoodDeckSF.dtype
checknull('WoodDeckSF')
plot_histogram_train_test('WoodDeckSF')
plot_kde_train_test('WoodDeckSF')
plot_ecdf_train_test('WoodDeckSF')
sns.lmplot('WoodDeckSF', 'SalePrice', train_df)
train_df.OpenPorchSF.dtype
checknull('OpenPorchSF')
plot_histogram_train_test('OpenPorchSF')
plot_kde_train_test('OpenPorchSF')
plot_ecdf_train_test('OpenPorchSF')
sns.lmplot('OpenPorchSF', 'SalePrice', train_df)
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVC



from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_log_error
combine = [train_df, test_df]
feature_cols = ['LotFrontage', 'OverallQual', 'YearBuilt',

       'TotalBsmtSF', '1stFlrSF', 'GrLivArea',

       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',

       'GarageArea']
train_df = train_df[feature_cols+['SalePrice']]

test_df = test_df[feature_cols]
# Impute columns with missing value

for col in feature_cols:

    feature_most_frequent_value_train = train_df[col].mode().values[0]

    train_df[col] = train_df[col].fillna(feature_most_frequent_value_train)

    test_df[col] = test_df[col].fillna(feature_most_frequent_value_train)
labels = train_df['SalePrice']

data = train_df.drop('SalePrice', axis=1)



mean_vec = np.mean(data, axis=0)

std_vec = np.std(data, axis=0)



data = (data - mean_vec) / std_vec



test_data = (test_df-mean_vec)/std_vec
X_train, X_test, y_train, y_test = train_test_split(data,

                                                   train_df['SalePrice'],

                                                   test_size=0.2,

                                                   random_state=123)
# params = {'C':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]}



# grid = GridSearchCV(param_grid=params, estimator=LogisticRegression(multi_class='auto', solver='lbfgs'), cv=5, n_jobs=-1)
# grid.fit(X_train, y_train)
lr = LogisticRegression(multi_class='auto', solver='lbfgs')

# lr.fit(X_train, y_train)
dt = DecisionTreeRegressor(max_depth=9, min_samples_leaf=20)

# dt.fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=50, max_depth=9 ,min_samples_leaf=5)

# rf.fit(X_train, y_train)
gb = GradientBoostingRegressor(n_estimators=80)

gb.fit(X_train, y_train)
y_pred_train = gb.predict(X_train)

mean_squared_log_error(y_train, y_pred_train)**.5
y_pred_val = gb.predict(X_test)

mean_squared_log_error(y_test, y_pred_val)**.5
gb.fit(np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]))

y_pred_test = gb.predict(test_data)
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission['SalePrice'] = y_pred_test
submission.to_csv('submission.csv', index=False)
submission.head()