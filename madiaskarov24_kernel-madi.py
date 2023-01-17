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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



sns.set_style('darkgrid')
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
train.head()
test.head()
submission.tail()
print("The train data size is : {} ".format(train.shape))

print("The test data size is : {} ".format(test.shape))
train.columns
train.info()
train.tail()
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop('Id', axis = 1, inplace = True)

test.drop('Id', axis = 1, inplace = True)
#check again the data size after dropping the 'Id' variable

print("The train data size is : {} ".format(train.shape)) 

print("The test data size is : {} ".format(test.shape))
train.describe()
train.corr()
fig, ax = plt.subplots()

ax.scatter(x = train['OverallQual'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('OverallQual', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)

plt.show()
train = train.drop(train[(train['TotalBsmtSF']>6000) & (train['SalePrice']<200000)].index)



fig, ax = plt.subplots()

ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)

plt.show()
from scipy import stats

from scipy.stats import norm, skew



sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
train["SalePrice"] = np.log1p(train["SalePrice"])
sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
train_features = train.drop(['SalePrice'], axis=1)

test_features = test

features = pd.concat([train_features, test_features]).reset_index(drop=True)
numeric_features = features.dtypes[features.dtypes != 'object'].index

numeric_features

len(numeric_features) 
category_features = features.dtypes[features.dtypes == 'object'].index

category_features

len(category_features)
#some columnsare not numeric

features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)
#Typ --> Typical Functionality

features['Functional'] = features['Functional'].fillna('Typ')

#SBrkr --> Standard Circuit Breakers & Romex

features['Electrical'] = features['Electrical'].fillna("SBrkr")

#TA --> Typical/Average

features['KitchenQual'] = features['KitchenQual'].fillna("TA")





features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



features["PoolQC"] = features["PoolQC"].fillna("None")

#None

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')

#None

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')

    

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)



features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) 

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])



features[numeric_features] = features[numeric_features].apply(

            lambda x: x.fillna(0))

features[category_features] = features[category_features].apply(

            lambda x: x.fillna('None'))
features.isnull().sum()
features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1) 
features_na = (features.isnull().sum() / len(features)) * 100

features_na = features_na.drop(features_na[features_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Data' : features_na})

missing_data.head()
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(features[c].values)) 

    features[c] = lbl.transform(list(features[c].values))



# shape        

print('Shape all_data: {}'.format(features.shape))
features.columns
#get dummies

features = pd.get_dummies(features)

print(features.shape)
train = features[:train.shape[0]]

test = features[test.shape[0]:]
y_train = train.SalePrice.values
from sklearn.linear_model import BayesianRidge

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)



model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)



rf = RandomForestRegressor(n_estimators=1200, max_depth=15,

                          min_samples_split=5,min_samples_leaf=5,

                          max_features=None,oob_score=True,

                          random_state=42)
score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score1 = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score1.mean(), score1.std()))
score2 = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score2.mean(), score2.std()))
score3 = rmsle_cv(rf)

print("Randome Forest score: {:.4f} ({:.4f})\n".format(score3.mean(), score3.std()))
model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
GBoost.fit(train, y_train)

GBoost_train_pred = GBoost.predict(train)

GBoost_pred = np.expm1(GBoost.predict(test))

print(rmsle(y_train, GBoost_train_pred))
model_lgb.fit(train, y_train)

lgb_train_pred = model_lgb.predict(train)

lgb_pred = np.expm1(model_lgb.predict(test.values))

print(rmsle(y_train, lgb_train_pred))
rf.fit(train, y_train)

rf_train_pred = rf.predict(train)

rf_pred = np.expm1(rf.predict(test.values))

print(rmsle(y_train, rf_train_pred))
dtypes = features.dtypes

cols_numeric = dtypes[dtypes != object].index.tolist()



# MSubClass should be treated as categorical

cols_numeric.remove('MSSubClass')



# choose any numeric column with less than 13 values to be

# "discrete". 13 chosen to include months of the year.

# other columns "continuous"

col_nunique = dict()



for col in cols_numeric:

    col_nunique[col] = features[col].nunique()

    

col_nunique = pd.Series(col_nunique)



cols_discrete = col_nunique[col_nunique<13].index.tolist()

cols_continuous = col_nunique[col_nunique>=13].index.tolist()



print(len(cols_numeric),'numeric columns, of which',

      len(cols_continuous),'are continuous and',

      len(cols_discrete),'are discrete.')
train_0 = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
features = pd.get_dummies(features)

print(features.shape)
fcols = 2

frows = len(cols_continuous)

plt.figure(figsize=(5*fcols,4*frows))



i=0

for col in cols_continuous:

    i+=1

    ax=plt.subplot(frows,fcols,i)

    sns.regplot(x=col, y=train_0['SalePrice'], data=train_0, ax=ax, 

                scatter_kws={'marker':'.','s':3,'alpha':0.3},

                line_kws={'color':'k'});

    plt.xlabel(col)

    plt.ylabel('SalePrice')

    

    i+=1

    ax=plt.subplot(frows,fcols,i)

    sns.distplot(features[col].dropna() , fit=stats.norm)

    plt.xlabel(col)
q1 = submission['SalePrice'].quantile(0.005)

q2 = submission['SalePrice'].quantile(0.995)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("my_submission.csv", index=False)