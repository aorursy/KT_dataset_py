import math

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import sklearn.metrics as metrics

from scipy import stats

from scipy.stats import norm, skew

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train
test
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
# Concatenate train and test data to make c

data = pd.concat([train, test])
data.set_index('Id', inplace = True)
data.isnull().sum()[:50]
data.info()
data['PoolQC'].fillna('None', inplace=True) # NaN values mean 'No Pool'

data['MiscFeature'].fillna('None', inplace=True) # NaN values mean 'No MiscFeature'

data['Alley'].fillna('None', inplace=True) # NaN values mean 'No alley access'

data['Fence'].fillna('None', inplace=True) # NaN values mean 'No Fence'

data['FireplaceQu'].fillna('None', inplace=True) # NaN values mean 'No Fireplace'
# GSince the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood, we can fill in missing values by the median LotFrontage of the neiborhood.

data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# GarageType, GarageFinish, GarageQual and GarageCond: Replacing missing data with None

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    data[col] = data[col].fillna('None')
# GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    data[col] = data[col].fillna(0)
# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    data[col] = data[col].fillna(0)
# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    data[col] = data[col].fillna('None')
# MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type. 

data["MasVnrType"] = data["MasVnrType"].fillna("None")

data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
# MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL

data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
# Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.

data = data.drop(['Utilities'], axis=1)
# Functional : data description says NA means typical

data["Functional"] = data["Functional"].fillna("Typ")
# Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.

data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.

data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
# Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string

data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
# SaleType : Fill in again with most frequent which is "WD"

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
# MSSubClass : Na most likely means No building class. We can replace missing values with None

data['MSSubClass'] = data['MSSubClass'].fillna("None")
data.info()
# Transforming some numerical variables that are really categorical



#MSSubClass=The building class

data['MSSubClass'] = data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

data['OverallCond'] = data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)
# Label Encoding some categorical variables that may contain information in their ordering set



from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(data[c].values)) 

    data[c] = lbl.transform(list(data[c].values))



# shape        

print('Shape data: {}'.format(data.shape))
data = pd.get_dummies(data)

print(data.shape)
train = data[:1460]

test = data[1460:].drop('SalePrice', axis = 1)
train.shape, test.shape
X = train.drop('SalePrice', axis = 1)

y = train['SalePrice']
# Split in train and validation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
xgb = XGBRegressor(booster='gbtree', colsample_bylevel=1,

                   colsample_bynode=1,colsample_bytree=0.6,

                   gamma=0, importance_type='gain',

                   learning_rate=0.01, max_delta_step=0,

                   max_depth=4,min_child_weight=1.5,

                   n_estimators=2500, n_jobs=1, nthread=None,

                   objective='reg:linear', reg_alpha=0.4640,

                   reg_lambda=0.6, scale_pos_weight=1,

                   silent=None, subsample=0.8, verbosity=1)
lgbm = LGBMRegressor(objective='regression',

                    num_leaves=4,

                    learning_rate=0.01,

                    n_estimators=11000,

                    max_bin=200,

                    bagging_fraction=0.75,

                    bagging_freq=5,

                    bagging_seed=7,

                    feature_fraction=0.4)
gboost = GradientBoostingRegressor(n_estimators=3000, 

                                   learning_rate=0.05,

                                   max_depth=4, 

                                   max_features='sqrt',

                                   min_samples_leaf=15, 

                                   min_samples_split=10, 

                                   loss='huber', 

                                   random_state =5)
xgb.fit(X_train, y_train)

lgbm.fit(X_train, y_train, eval_metric='rmsle')

gboost.fit(X_train, y_train)
pred2 = xgb.predict(X_test)

pred3 = lgbm.predict(X_test)

pred4 = gboost.predict(X_test)
print('Root Mean Square Logarithmic Error test (XGB) = ' + str(math.sqrt(metrics.mean_squared_log_error(y_test, pred2))))

print('Root Mean Square Logarithmic Error test (LGBM) = ' + str(math.sqrt(metrics.mean_squared_log_error(y_test, pred3))))

print('Root Mean Square Logarithmic Error test (GBoost) = ' + str(math.sqrt(metrics.mean_squared_log_error(y_test, pred4))))
lgbm.fit(X, y)   # 0.12269 

xgb.fit(X ,y)    # 0.12495

gboost.fit(X, y) # 0.12333
prediction_lgbm =  np.expm1(lgbm.predict(test))

prediction_xgb = np.expm1(xgb.predict(test))

prediction_gboost = np.expm1(gboost.predict(test))
"""

prediction = ( prediction_lgbm * 0.38 + prediction_gboost * 0.35 + prediction_xgb * 0.27)   # 0.12006

prediction = ( prediction_lgbm * 0.4 + prediction_gboost * 0.35 + prediction_xgb * 0.25)    # 0.12007

prediction = ( prediction_lgbm * 0.45 + prediction_gboost * 0.35 + prediction_xgb * 0.2)    # 0.12012

prediction = ( prediction_lgbm * 0.55 + prediction_gboost * 0.45)                           # 0.12061

prediction = ( prediction_lgbm * 0.45 + prediction_gboost * 0.55)                           # 0.12069

prediction = ( prediction_gboost * 0.15 + prediction_lgbm * 0.7 + prediction_gboost * 0.15) # 0.12086

prediction = ( prediction_gboost * 0.2 + prediction_lgbm * 0.5 + prediction_gboost * 0.3)   # 0.12154

prediction = ( prediction_lgbm * 0.55 + prediction_xgb * 0.45)                              # 0.12155

"""
prediction = ( prediction_lgbm * 0.38 + prediction_gboost * 0.35 + prediction_xgb * 0.27)   # 0.12006
submission = pd.DataFrame({"Id": test.index,"SalePrice": prediction})
submission.to_csv('submission.csv', index=False)