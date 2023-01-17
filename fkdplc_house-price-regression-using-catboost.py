import numpy as np

import pandas as pd

import os

import math

from tqdm.notebook import tqdm

from sklearn.exceptions import ConvergenceWarning

import warnings

from matplotlib import pyplot as plt

import seaborn as sns

from scipy import stats as scs



warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=ConvergenceWarning)



PATH = '/kaggle/input/house-prices-advanced-regression-techniques/'

train = pd.read_csv(PATH + 'train.csv')

test = pd.read_csv(PATH + 'test.csv')



train.head()
# show which columns have nans

def show_nans():

    nas = train.isna().sum() + test.isna().sum()

    return nas[nas != 0]



# plots numeric feature

def explore_cont(feature, data=train, kind='reg', order=1):

    plt.figure(1, figsize=(10, 10))

    sns.jointplot(x=feature, y='SalePrice', data=data, kind=kind, order=order)

    plt.show()

    

# plots categorical feature

def explore_cat(feature, data=train, kind='reg'):

    plt.figure(1, figsize=(10, 10))

    sns.violinplot(x=feature, y='SalePrice', data=data, bw=.2)

    plt.show()



# drops feature from both sets

def drop(col):

    train.drop(columns=[col], inplace=True)

    test.drop(columns=[col], inplace=True)



# used for creating new feature

def apply(col, new_col, func, drop_col=False):

    train[new_col] = train[col].apply(func)

    test[new_col] = test[col].apply(func)

    if drop_col:

        drop(col)

    

# fill nans

def fillna(col, fill_with='NA'):

    train[col].fillna(fill_with, inplace=True)

    test[col].fillna(fill_with, inplace=True)



# plots histogram

def show_hist(values):

    plt.figure(1, figsize=(10, 6))

    sns.distplot(values)

    plt.show()

    

    print('skew:', scs.skew(values))

    print('kurtosis:', scs.kurtosis(values))



# define target variable conversions

target_trans = lambda price: np.log1p(price) ** .5

target_inv_trans = lambda price: np.expm1(price ** 2)



# convert

train['SalePrice'] = target_trans(train['SalePrice'])



# visualize

show_hist(train['SalePrice'])
show_nans()
explore_cont('LotArea')
explore_cont('GrLivArea')
explore_cont('TotalBsmtSF')
# categorical that support 'NA'

features_cat_with_na = [

    'Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 

    'BsmtQual', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageFinish', 

    'GarageQual', 'GarageType',  'MiscFeature', 'PoolQC', 

]



# numerical that have NaNs

features_cont_with_na = [

    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath',  'BsmtUnfSF', 

    'BsmtHalfBath', 'GarageArea', 'GarageCars',  'GarageYrBlt',

    'LotFrontage', 'MasVnrArea', 'TotalBsmtSF', 

]



# possibly being removed

features_with_too_much_nas = [

    'MiscFeature', 'PoolQC', 'Fence', 'FireplaceQu', 'Alley'

]



for feat in features_cat_with_na:

    fillna(feat)

    

for feat in features_cont_with_na:

    fillna(feat, train[feat].mean())



# these features don't support 'NA', they have different values

fillna('Electrical', 'SBrkr')

fillna('Exterior1st', 'Other')

fillna('Exterior2nd', 'Other')

fillna('Functional', 'Typ')

fillna('KitchenQual', 'TA')

fillna('SaleType', 'Oth')

fillna('MasVnrType', 'None')



# just with mode

fillna('MSZoning', train['MSZoning'].mode()[0])

fillna('Utilities', train['MSZoning'].mode()[0])



# remove outliers based on plots above

train = train[train['LotArea'] < 30000]

train = train[train['GrLivArea'] < 4000]

train = train[train['TotalBsmtSF'] < 2800]



# extract target and ids for test set

target = train['SalePrice']

test_ids = test['Id']



# remove target and ids

train.drop(columns=['SalePrice', 'Id'], inplace=True)

test.drop(columns=['Id'], inplace=True)



train.head()
# extract categorical features names for CatBoost

cat_features = list(train.select_dtypes(exclude=['int', 'float']).columns.values)

len(cat_features)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_log_error as msle

from sklearn.metrics import make_scorer

from catboost import CatBoostRegressor



# split train set

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=289)



# make score function for GridSearchCV

def score_func(y_true, y_pred, **kwargs):

    return msle(target_inv_trans(y_true), target_inv_trans(y_pred), **kwargs) ** .5



# hyperparams setting

def make_search(estimator, params, verbose=1):

    scorer = make_scorer(score_func, greater_is_better=False)

    search = GridSearchCV(estimator, params, cv=5, scoring=scorer, verbose=11, n_jobs=-1)

    search.fit(x_train, y_train)

    results = pd.DataFrame()

    for k, v in search.cv_results_.items():

        results[k] = v

    results = results.sort_values(by='rank_test_score')

    best_params_row = results[results['rank_test_score'] == 1]

    mean, std = best_params_row['mean_test_score'].iloc[0], best_params_row['std_test_score'].iloc[0]

    best_params = best_params_row['params'].iloc[0]

    if verbose:

        print('%s: %.4f (%.4f) with params' % (estimator.__class__.__name__, -mean, std), best_params)

    return best_params



depths = list(range(2, 7))

estimators = [1800, 2000, 3000]



# i calculated them earlier

best_params = {

     'n_estimators': 3000,

    'max_depth': 4,

    'random_state': 289,

    'cat_features': cat_features,

    'verbose': False

}



# pass True to rerun search

if False:

    search_params = {

        'n_estimators': estimators,

        'max_depth': depths,

        'random_state': [289],

        'cat_features': [cat_features],

        'verbose': [False]

    }

    best_params = make_search(CatBoostRegressor(), search_params)





# fitting best model

model = CatBoostRegressor()

model.set_params(**best_params)

model.fit(x_train, y_train)



y_true = target_inv_trans(y_test)

y_pred = target_inv_trans(model.predict(x_test))

print('msle = %.4f' % msle(y_true, y_pred) ** .5)
res = pd.DataFrame()

res['Id'] = test_ids

res['SalePrice'] = target_inv_trans(model.predict(test))

res.to_csv('submission.csv', index=False)

res.head(20)