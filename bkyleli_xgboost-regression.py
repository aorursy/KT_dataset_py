# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#Keep ID column

train_ID = train['Id']

test_ID = test['Id']
#Drop ID

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
#Check outliers

plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#Drop outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
#Log target values

train["SalePrice"] = np.log1p(train["SalePrice"])
#Feature engineering preperation

ntrain = train.shape[0]

y_train = train.SalePrice.values

concat_data = pd.concat((train, test)).reset_index(drop=True)

concat_data.drop(['SalePrice'], axis=1, inplace=True)
#Check missing values

concat_data.isnull().sum().sort_values(ascending=False)
#Fill missing values 

concat_data["PoolQC"] = concat_data["PoolQC"].fillna("None")

concat_data["MiscFeature"] = concat_data["MiscFeature"].fillna("None")

concat_data["Alley"] = concat_data["Alley"].fillna("None")

concat_data["Fence"] = concat_data["Fence"].fillna("None")

concat_data["FireplaceQu"] = concat_data["FireplaceQu"].fillna("None")
#Fill missing values

concat_data["LotFrontage"] = concat_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
#Fill missing values

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    concat_data[col] = concat_data[col].fillna('None')
#Fill missing values

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    concat_data[col] = concat_data[col].fillna(0)
#Fill missing values

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    concat_data[col] = concat_data[col].fillna(0)
#Fill missing values

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    concat_data[col] = concat_data[col].fillna('None')
#Fill missing values

concat_data["MasVnrType"] = concat_data["MasVnrType"].fillna("None")

concat_data["MasVnrArea"] = concat_data["MasVnrArea"].fillna(0)
#Fill missing values

concat_data['MSZoning'] = concat_data['MSZoning'].fillna(concat_data['MSZoning'].mode()[0])
#Fill missing values

concat_data = concat_data.drop(['Utilities'], axis=1)
#Fill missing values

concat_data["Functional"] = concat_data["Functional"].fillna("Typ")
#Fill missing values

concat_data['KitchenQual'] = concat_data['KitchenQual'].fillna(concat_data['KitchenQual'].mode()[0])
#Fill missing values

concat_data['Exterior1st'] = concat_data['Exterior1st'].fillna(concat_data['Exterior1st'].mode()[0])

concat_data['Exterior2nd'] = concat_data['Exterior2nd'].fillna(concat_data['Exterior2nd'].mode()[0])
#Fill missing values

concat_data['SaleType'] = concat_data['SaleType'].fillna(concat_data['SaleType'].mode()[0])
#Fill missing values

concat_data['MSSubClass'] = concat_data['MSSubClass'].fillna("None")
#Change dtype

concat_data['MSSubClass'] = concat_data['MSSubClass'].apply(str)

concat_data['OverallCond'] = concat_data['OverallCond'].astype(str)

concat_data['YrSold'] = concat_data['YrSold'].astype(str)

concat_data['MoSold'] = concat_data['MoSold'].astype(str)
#LabelEncoder to categorical features

from sklearn.preprocessing import LabelEncoder

CatFeatures = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond','HeatingQC', 

               'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 

               'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 

               'OverallCond', 'YrSold', 'MoSold')



lbl = LabelEncoder() 



for col in CatFeatures:

    lbl.fit(list(concat_data[col].values)) 

    concat_data[col] = lbl.transform(list(concat_data[col].values))
# Add total sqfootage feature 

concat_data['TotalSF'] = concat_data['TotalBsmtSF'] + concat_data['1stFlrSF'] + concat_data['2ndFlrSF']
#Get dummies

concat_data = pd.get_dummies(concat_data)

print(concat_data.shape)
#Split train test dataset

train = concat_data[:ntrain]

test = concat_data[ntrain:]
#Tune parameters

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error



#parameters = {'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],

#              'gamma':[0, 0.025, 0.05, 0.075, 0.1],

#              'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],

#             'max_depth': [3, 5, 10, 15, 20],

#             'min_child_weight': [0, 2, 5, 10, 20],

#             'n_estimators': [500, 1000, 2000, 3000, 5000],

#             'reg_alpha': [0, 0.25, 0.5, 0.75, 1],

#             'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],

#             'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],

#            }





#model_xgb = xgb.XGBRegressor(colsample_bytree=0.5, gamma=0.05, 

#                            learning_rate=0.05, max_depth=3, 

#                            min_child_weight=5, n_estimators=2000,

#                            reg_alpha=0.5, reg_lambda=0.8,

#                            subsample=0.5, random_state =7)



#gsearch = GridSearchCV(model_xgb, param_grid=parameters, scoring='neg_mean_squared_error', cv=3)

#gsearch.fit(train, y_train)



#gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
#Best model

model_xgb = xgb.XGBRegressor(colsample_bytree=0.5, gamma=0.05, 

                            learning_rate=0.05, max_depth=3, 

                            min_child_weight=2, n_estimators=2000,

                            reg_alpha=0.5, reg_lambda=0.8,

                            subsample=0.5, random_state =7)
#Fit model

model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))
#Check rmse

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(y_train, xgb_train_pred)))
#Submit

sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = xgb_pred

sub.to_csv('submission.csv',index=False)