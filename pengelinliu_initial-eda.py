# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

fulldata=[train,test]
train.head(5)

print(test.info())
NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])

NAs.head(5)

NAs[NAs.sum(axis=1)>0]
def get_score(prediction, labels):

    print('R2: {}'.format(r2_score(prediction, lables)))# format the value in string {}

    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables)))) # 
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
train_labels=train.pop('SalePrice')  #  remove SalePrice and add to the train_labels,  'Series'

features=pd.concat([train,test],keys=['train','test'])

train_labels
features.head(5)
features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',

               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',

               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],

              axis=1, inplace=True)

features.head(5)
# MSSubClass as str

features['MSSubClass'] = features['MSSubClass'].astype(str)



# MSZoning NA in pred. filling with most popular values

features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])



# LotFrontage  NA in all. I suppose NA means 0

features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())



# Alley  NA in all. NA means no access

features['Alley'] = features['Alley'].fillna('NOACCESS')



# Converting OverallCond to str

features.OverallCond = features.OverallCond.astype(str)



# MasVnrType NA in all. filling with most popular values

features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])



# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2

# NA in all. NA means No basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('NoBSMT')



# TotalBsmtSF  NA in pred. I suppose NA means 0

features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)



# Electrical NA in pred. filling with most popular values

features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])



# KitchenAbvGr to categorical

features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str)



# KitchenQual NA in pred. filling with most popular values

features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])



# FireplaceQu  NA in all. NA means No Fireplace

features['FireplaceQu'] = features['FireplaceQu'].fillna('NoFP')



# GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage

for col in ('GarageType', 'GarageFinish', 'GarageQual'):

    features[col] = features[col].fillna('NoGRG')



# GarageCars  NA in pred. I suppose NA means 0

features['GarageCars'] = features['GarageCars'].fillna(0.0)



# SaleType NA in pred. filling with most popular values

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])



# Year and Month to categorical

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)



# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features

features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']

features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
ax = sns.distplot(train_labels, kde=False)

# a little shewed right
train_labels = np.log(train_labels)
ax = sns.distplot(train_labels, kde=False)
numeric_features=features.loc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]  #  slice row and columns

numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()
ax = sns.pairplot(numeric_features_standardized)

# Getting Dummies from Condition1 and Condition2

conditions = set([x for x in features['Condition1']] + [x for x in features['Condition2']]) # string adding

dummies = pd.DataFrame(data=np.zeros((len(features.index), len(conditions))),

                       index=features.index, columns=conditions)       ## set 0 to Dataframe

for i, cond in enumerate(zip(features['Condition1'], features['Condition2'])):

    dummies.ix[i, cond] = 1 # i position, cond colummns

features = pd.concat([features, dummies.add_prefix('Condition_')], axis=1)



exteriors = set([x for x in features['Exterior1st']] + [x for x in features['Exterior2nd']])

dummies = pd.DataFrame(data=np.zeros((len(features.index), len(exteriors))),

                       index=features.index, columns=exteriors)

for i, ext in enumerate(zip(features['Exterior1st'], features['Exterior2nd'])):

    dummies.ix[i, ext] = 1

features = pd.concat([features, dummies.add_prefix('Exterior_')], axis=1)

features.drop(['Exterior1st', 'Exterior2nd', 'Exterior_nan'], axis=1, inplace=True)



features





features.dtypes[features.dtypes == 'object'].index
for col in features.dtypes[features.dtypes == 'object'].index:

    for_dummy = features.pop(col)  # pop columns into for_dummy

    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)

features
features_standardized = features.copy()

features_standardized.update(numeric_features_standardized)
from sklearn import ensemble, tree, linear_model

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle

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



   
train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

# including only numeric numbers







train_features_st = features_standardized.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

test_features_st = features_standardized.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

# including only numeric numbers



x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)

x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, train_labels, test_size=0.1, random_state=200)