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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
sm_train = pd.read_csv('../input/train.csv')
sm_train.columns
sm_train['SalePrice'].describe()
#histogram

sns.distplot(sm_train['SalePrice']);
#skewness and kurtosis

print("Skewness: %f" % sm_train['SalePrice'].skew())

print("Kurtosis: %f" % sm_train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([sm_train['SalePrice'], sm_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([sm_train['SalePrice'], sm_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([sm_train['SalePrice'], sm_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([sm_train['SalePrice'], sm_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
#correlation matrix

corrmat = sm_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(sm_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(sm_train[cols], size = 2.5)

plt.show();
#missing data

total = sm_train.isnull().sum().sort_values(ascending=False)

percent = (sm_train.isnull().sum()/sm_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#dealing with missing data

sm_train = sm_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

sm_train = sm_train.drop(sm_train.loc[sm_train['Electrical'].isnull()].index)

sm_train.isnull().sum().max() #just checking that there's no missing data missing...
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(sm_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([sm_train['SalePrice'], sm_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

sm_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

sm_train = sm_train.drop(sm_train[sm_train['Id'] == 1299].index)

sm_train = sm_train.drop(sm_train[sm_train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([sm_train['SalePrice'], sm_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#histogram and normal probability plot

sns.distplot(sm_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(sm_train['SalePrice'], plot=plt)
#applying log transformation

sm_train['SalePrice'] = np.log(sm_train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(sm_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(sm_train['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(sm_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(sm_train['GrLivArea'], plot=plt)
#data transformation

sm_train['GrLivArea'] = np.log(sm_train['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(sm_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(sm_train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(sm_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(sm_train['TotalBsmtSF'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

sm_train['HasBsmt'] = pd.Series(len(sm_train['TotalBsmtSF']), index=sm_train.index)

sm_train['HasBsmt'] = 0 

sm_train.loc[sm_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

sm_train.loc[sm_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(sm_train['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(sm_train[sm_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(sm_train[sm_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#scatter plot

plt.scatter(sm_train['GrLivArea'], sm_train['SalePrice']);
#scatter plot

plt.scatter(sm_train[sm_train['TotalBsmtSF']>0]['TotalBsmtSF'], sm_train[sm_train['TotalBsmtSF']>0]['SalePrice']);
#convert categorical variable into dummy

sm_train = pd.get_dummies(sm_train)
from sklearn.ensemble import RandomForestRegressor



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(5)
test.head(5)
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(train.shape))

print("The test data size before dropping Id feature is : {} ".format(test.shape))



#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test.shape))
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
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
# Spliting to features and lables and deleting variable I don't need

train_labels = train.pop('SalePrice')



features = pd.concat([train, test], keys=['train', 'test'])



# I decided to get rid of features that have more than half of missing information or do not correlate to SalePrice

features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',

               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',

               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],

              axis=1, inplace=True)
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
## Standardizing numeric features

numeric_features = features.loc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]

numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()
ax = sns.pairplot(numeric_features_standardized)
### Copying features

features_standardized = features.copy()



### Replacing numeric features by standardized values

features_standardized.update(numeric_features_standardized)
from sklearn import ensemble, tree, linear_model

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle

### Splitting features

train_features = features.loc['train'].select_dtypes(include=[np.number]).values

test_features = features.loc['test'].select_dtypes(include=[np.number]).values



### Splitting standardized features

train_features_st = features_standardized.loc['train'].select_dtypes(include=[np.number]).values

test_features_st = features_standardized.loc['test'].select_dtypes(include=[np.number]).values
### Shuffling train sets

train_features_st, train_features, train_labels = shuffle(train_features_st, train_features, train_labels, random_state = 5)
### Splitting

x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)

x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, train_labels, test_size=0.1, random_state=200)
ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train_st, y_train_st)

train_test(ENSTest, x_train_st, x_test_st, y_train_st, y_test_st)
# Average R2 score and standart deviation of 5-fold cross-validation

scores = cross_val_score(ENSTest, train_features_st, train_labels, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',

                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)

train_test(GBest, x_train, x_test, y_train, y_test)
# Average R2 score and standart deviation of 5-fold cross-validation

scores = cross_val_score(GBest, train_features_st, train_labels, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Retraining models

GB_model = GBest.fit(train_features, train_labels)

ENST_model = ENSTest.fit(train_features_st, train_labels)
## Getting our SalePrice estimation

Final_labels = (np.exp(GB_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2
## Saving to CSV

pd.sm_train({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('2019-02-28.csv', index =False) 