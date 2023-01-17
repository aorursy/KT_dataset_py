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



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
data = pd.concat([train_df, test_df], sort=False)

data = data.reset_index(drop=True)

data.head()
# Missing Values

nans=pd.isnull(data).sum()

nans[nans>0]
data['MSZoning']  = data['MSZoning'].fillna(data['MSZoning'].mode()[0])

data['Utilities'] = data['Utilities'].fillna(data['Utilities'].mode()[0])

data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])



data["BsmtFinSF1"]  = data["BsmtFinSF1"].fillna(0)

data["BsmtFinSF2"]  = data["BsmtFinSF2"].fillna(0)

data["BsmtUnfSF"]   = data["BsmtUnfSF"].fillna(0)

data["TotalBsmtSF"] = data["TotalBsmtSF"].fillna(0)

data["BsmtFullBath"] = data["BsmtFullBath"].fillna(0)

data["BsmtHalfBath"] = data["BsmtHalfBath"].fillna(0)

data["BsmtQual"] = data["BsmtQual"].fillna("None")

data["BsmtCond"] = data["BsmtCond"].fillna("None")

data["BsmtExposure"] = data["BsmtExposure"].fillna("None")

data["BsmtFinType1"] = data["BsmtFinType1"].fillna("None")

data["BsmtFinType2"] = data["BsmtFinType2"].fillna("None")



data['KitchenQual']  = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])

data["Functional"]   = data["Functional"].fillna("Typ")

data["FireplaceQu"]  = data["FireplaceQu"].fillna("None")



data["GarageType"]   = data["GarageType"].fillna("None")

data["GarageYrBlt"]  = data["GarageYrBlt"].fillna(0)

data["GarageFinish"] = data["GarageFinish"].fillna("None")

data["GarageCars"] = data["GarageCars"].fillna(0)

data["GarageArea"] = data["GarageArea"].fillna(0)

data["GarageQual"] = data["GarageQual"].fillna("None")

data["GarageCond"] = data["GarageCond"].fillna("None")



data["PoolQC"] = data["PoolQC"].fillna("None")

data["Fence"]  = data["Fence"].fillna("None")

data["MiscFeature"] = data["MiscFeature"].fillna("None")

data['SaleType']    = data['SaleType'].fillna(data['SaleType'].mode()[0])

data['LotFrontage'].interpolate(method='linear',inplace=True)

data["Electrical"]  = data.groupby("YearBuilt")['Electrical'].transform(lambda x: x.fillna(x.mode()[0]))

data["Alley"] = data["Alley"].fillna("None")



data["MasVnrType"] = data["MasVnrType"].fillna("None")

data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
nans=pd.isnull(data).sum()

nans[nans>0]
from sklearn import preprocessing

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.metrics import r2_score

from sklearn.ensemble import StackingRegressor

from sklearn.linear_model import ElasticNet, Lasso

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler



import lightgbm as lgbm

import xgboost as xgb



import warnings

warnings.filterwarnings(action='ignore')
_list = []

for col in data.columns:

    if type(data[col][0]) == type('str'): 

        _list.append(col)



le = preprocessing.LabelEncoder()

for li in _list:

    le.fit(list(set(data[li])))

    data[li] = le.transform(data[li])
train, test = data[:len(train_df)], data[len(train_df):]



X = train.drop(columns=['SalePrice', 'Id']) 

y = train['SalePrice']



test = test.drop(columns=['SalePrice', 'Id'])
kfold = KFold(n_splits=10, random_state = 77, shuffle = True)
# LightGBM Grid Search

params = {

    'task' : 'train',

    'objective' : 'regression',

    'subsample' : 0.8,

    'max_depth' : 7

}



param_grid = {

    'learning_rate': [0.1],

    'feature_fraction' : [0.5, 0.8],

    'num_leaves':[31, 63, 127]

}



lgbm_model = lgbm.LGBMRegressor(**params, verbose=-1)



lgbm_grid  = GridSearchCV(lgbm_model, 

                          param_grid, 

                          cv=kfold, 

                          scoring='neg_mean_squared_error', 

                          return_train_score=True)



lgbm_grid.fit(X, y)



r2_score(lgbm_grid.predict(X), y)
lgbm_grid.best_estimator_
# XGB Grid Search

param_grid = {

    'learning_rate': [0.1],

    'subsample' : [0.5, 0.8, 1.0],

    'max_depth':[4, 5, 8]

}



xgb_model = xgb.XGBRegressor()



xgb_grid  = GridSearchCV(xgb_model, 

                          param_grid, 

                          cv=kfold, 

                          scoring='neg_mean_squared_error')



xgb_grid.fit(X, y)



r2_score(xgb_grid.predict(X), y)
xgb_grid.best_estimator_
# Lasso R

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0001))

lasso.fit(X, y)



r2_score(lasso.predict(X), y)
# ElasticNet R

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0001, l1_ratio=.9))

ENet.fit(X, y)



r2_score(ENet.predict(X), y)
# Stacking

stacking = StackingRegressor(estimators=[ 

    ('lgbm', lgbm_grid.best_estimator_),

    ('xgb', xgb_grid.best_estimator_),

    ('lasso', lasso.steps[1][1]),

    ('elastic',ENet.steps[1][1])],

                             final_estimator=lgbm_grid.best_estimator_,

                             passthrough=True)



model = stacking.fit(X, y)

r2_score(model.predict(X), y)
# Predict & Submit

submission = pd.DataFrame()

submission['Id'] = test_df['Id']

submission['SalePrice'] = np.round(list(model.predict(test)), 0)

submission.head()
submission.to_csv("submission.csv", index=False)