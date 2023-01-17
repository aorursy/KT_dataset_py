# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# get home price train & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv")

test    = pd.read_csv("../input/test.csv")

full = train.append(test, ignore_index=True)

print (train.shape, test.shape, full.shape)
train.head()
sns.lmplot(x='GrLivArea', y='SalePrice',  data=train)
train = train[train.GrLivArea < 4500]
sns.lmplot(x='GrLivArea', y='SalePrice',  data=train)
#Checking for missing data

NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])

NAs[NAs.sum(axis=1) > 0]
# Spliting to features and lables

train_labels = train.pop('SalePrice')



features = pd.concat([train, test], keys=['train', 'test'])
# Deleting features that are more than 50% missing

features.drop(['PoolQC', 'MiscFeature', 'FireplaceQu', 'Fence', 'Alley'],

              axis=1, inplace=True)

features.shape
# MSZoning NA in pred. filling with most popular values

features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])



# LotFrontage  NA in all. I suppose NA means 0

features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())



# MasVnrType NA in all. filling with most popular values

features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])



# MasVnrArea NA in all. filling with mean value

features['MasVnrArea'] = features['MasVnrArea'].fillna(features['MasVnrArea'].mean())



# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2

# NA in all. NA means No basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('NoBSMT')



# BsmtFinSF1 and BsmtFinSF2  NA in pred. I suppose NA means 0

features['BsmtFinSF1'] = features['BsmtFinSF1'].fillna(0)   

features['BsmtFinSF2'] = features['BsmtFinSF2'].fillna(0)  

    

# BsmtFullBath and BsmtHalfBath NA in all. filling with most popular value

features['BsmtFullBath'] = features['BsmtFullBath'].fillna(features['BsmtFullBath'].median())

features['BsmtHalfBath'] = features['BsmtHalfBath'].fillna(features['BsmtHalfBath'].median())



# BsmtUnfSF NA in all. Filling with mean value

features['BsmtUnfSF'] = features['BsmtUnfSF'].fillna(features['BsmtUnfSF'].mean())



# Exterior1st and Exterior2nd NA in all. filling with most popular value

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])



# Functional NA in all. filling with most popular value

features['Functional'] = features['Functional'].fillna(features['Functional'].mode()[0])



# TotalBsmtSF  NA in pred. I suppose NA means 0

features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)



# Electrical NA in pred. filling with most popular values

features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])



# KitchenQual NA in pred. filling with most popular values

features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])



# GarageArea NA in all. NA means no garage so 0

features['GarageArea'] = features['GarageArea'].fillna(0.0)



# GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageQual', 'GarageCond'):

    features[col] = features[col].fillna('NoGRG')



# GarageCars  NA in pred. I suppose NA means 0

features['GarageCars'] = features['GarageCars'].fillna(0.0)



# SaleType NA in pred. filling with most popular values

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])



# Utilities NA in all. filling with most popular value

features['Utilities'] = features['Utilities'].fillna(features['Utilities'].mode()[0])



# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features

features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageYrBlt'], axis=1, inplace=True)
features.shape
# Our SalesPrice is skewed right (check plot below). I'm logtransforming it. 

ax = sns.distplot(train_labels)
## Log transformation of labels

train_labels = np.log(train_labels)
## Now it looks much better

ax = sns.distplot(train_labels)
def num2cat(x):

    return str(x)
features['MSSubClass_str'] = features['MSSubClass'].apply(num2cat)

features.pop('MSSubClass')

features.shape
# Getting Dummies from all other categorical vars

for col in features.dtypes[features.dtypes == 'object'].index:

    for_dummy = features.pop(col)

    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)
features.shape
features.head()
### Splitting features

train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
### Splitting

x_train, x_test, y_train, y_test = train_test_split(train_features,

                                                    train_labels,

                                                    test_size=0.1,

                                                    random_state=200)
# Prints R2 and RMSE scores

def get_score(prediction, lables):    

    print('R2: {}'.format(r2_score(prediction, lables)))

    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))



# Shows scores for train and validation sets    

def train_test(estimator, x_trn, x_tst, y_trn, y_tst):

    prediction_train = estimator.predict(x_trn)

    # Printing estimator

    print(estimator)

    # Printing train scores

    get_score(prediction_train, y_trn)

    prediction_test = estimator.predict(x_tst)

    # Printing test scores

    print("Test")

    get_score(prediction_test, y_tst)
GBR = GradientBoostingRegressor(n_estimators=12000,

            learning_rate=0.05, max_depth=3, max_features='sqrt',

            min_samples_leaf=15, min_samples_split=10, loss='huber')
GBR.fit(x_train, y_train)
train_test(GBR, x_train, x_test, y_train, y_test)
# Average R2 score and standart deviation of 5-fold cross-validation

scores = cross_val_score(GBR, train_features, train_labels, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso.fit(x_train, y_train)
train_test(lasso, x_train, x_test, y_train, y_test)
# Average R2 score and standart deviation of 5-fold cross-validation

scores = cross_val_score(lasso, train_features, train_labels, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
ENet.fit(x_train, y_train)
train_test(ENet, x_train, x_test, y_train, y_test)
# Average R2 score and standart deviation of 5-fold cross-validation

scores = cross_val_score(ENet, train_features, train_labels, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
def averagingModels(X, train, labels, models=[]):

    for model in models:

        model.fit(train, labels)

    predictions = np.column_stack([

        model.predict(X) for model in models

    ])

    return np.mean(predictions, axis=1)
test_y = averagingModels(test_features, train_features, train_labels, [GBR, lasso, ENet])

test_y = np.exp(test_y)
test_id = test.Id

test_submit = pd.DataFrame({'Id': test_id, 'SalePrice': test_y})

test_submit.shape

test_submit.head()

#test_submit.to_csv('house_price_pred_avg.csv', index=False)