# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.metrics import mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

allhouses = pd.concat([train,test], sort=False)



train.drop('Id',axis=1,inplace = True)

test.drop('Id', axis=1, inplace = True)



print(train.shape)

print(test.shape)



len_train = train.shape[0]

len_test = test.shape[0]



allhouses = pd.concat((train,test),sort=False).reset_index(drop = True)

allhouses.drop(['SalePrice'], axis=1, inplace = True)



#count of missing values

allhouses.isnull().sum().sort_values(ascending=False)
def missing_data(data):

    na_values = data.isnull().sum()

    missratio = (data.isnull().sum() / len(data)) * 100

    missing = pd.DataFrame({'Missing Ratio' :missratio,'Missing Values' :na_values})

    print(missing.head(20))



missing_data(allhouses)
#%% Fill the Missing Values

allhouses['PoolQC'] = allhouses['PoolQC'].fillna("None")

allhouses['MiscFeature'] = allhouses['MiscFeature'].fillna("None")

allhouses["Alley"] = allhouses["Alley"].fillna("None")

allhouses["Fence"] = allhouses["Fence"].fillna("None")

allhouses["FireplaceQu"] = allhouses["FireplaceQu"].fillna("None")

allhouses["MasVnrType"] = allhouses["MasVnrType"].fillna("None")

allhouses["GarageType"] = allhouses["GarageType"].fillna("None")

allhouses["GarageFinish"] = allhouses["GarageFinish"].fillna("None")

allhouses["GarageQual"] = allhouses["GarageQual"].fillna("None")

allhouses["GarageCond"] = allhouses["GarageCond"].fillna("None")

allhouses["BsmtCond"] = allhouses["BsmtCond"].fillna("None")

allhouses["BsmtExposure"] = allhouses["BsmtExposure"].fillna("None")

allhouses["BsmtQual"] = allhouses["BsmtQual"].fillna("None")

allhouses["BsmtFinType2"] = allhouses["BsmtFinType2"].fillna("None")

allhouses["BsmtFinType1"] = allhouses["BsmtFinType1"].fillna("None")



allhouses["GarageYrBlt"] = allhouses["GarageYrBlt"].fillna(0)

allhouses["GarageArea"] = allhouses["GarageArea"].fillna(0)

allhouses["GarageCars"] = allhouses["GarageCars"].fillna(0)

allhouses["BsmtHalfBath"] = allhouses["BsmtHalfBath"].fillna(0)

allhouses["BsmtFullBath"] = allhouses["BsmtFullBath"].fillna(0)

allhouses["TotalBsmtSF"] = allhouses["TotalBsmtSF"].fillna(0)

allhouses["BsmtUnfSF"] = allhouses["BsmtUnfSF"].fillna(0)

allhouses["BsmtFinSF2"] = allhouses["BsmtFinSF2"].fillna(0)

allhouses["BsmtFinSF1"] = allhouses["BsmtFinSF1"].fillna(0)

allhouses["MasVnrArea"] = allhouses["MasVnrArea"].fillna(0)



allhouses['MSZoning'] = allhouses['MSZoning'].fillna(allhouses['MSZoning'].mode()[0])

allhouses["Functional"] = allhouses["Functional"].fillna(allhouses['Functional'].mode()[0])

allhouses['Electrical'] = allhouses['Electrical'].fillna(allhouses['Electrical'].mode()[0])

allhouses['KitchenQual'] = allhouses['KitchenQual'].fillna(allhouses['KitchenQual'].mode()[0])

allhouses['Exterior1st'] = allhouses['Exterior1st'].fillna(allhouses['Exterior1st'].mode()[0])

allhouses['Exterior2nd'] = allhouses['Exterior2nd'].fillna(allhouses['Exterior2nd'].mode()[0])

allhouses['SaleType'] = allhouses['SaleType'].fillna(allhouses['SaleType'].mode()[0])



allhouses["LotFrontage"] = allhouses.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



train['PoolQC'] = train['PoolQC'].fillna("None")

train['MiscFeature'] = train['MiscFeature'].fillna("None")

train["Alley"] = train["Alley"].fillna("None")

train["Fence"] = train["Fence"].fillna("None")

train["FireplaceQu"] = train["FireplaceQu"].fillna("None")

train["MasVnrType"] = train["MasVnrType"].fillna("None")

train["GarageType"] = train["GarageType"].fillna("None")

train["GarageFinish"] = train["GarageFinish"].fillna("None")

train["GarageQual"] = train["GarageQual"].fillna("None")

train["GarageCond"] = train["GarageCond"].fillna("None")

train["BsmtCond"] = train["BsmtCond"].fillna("None")

train["BsmtExposure"] = train["BsmtExposure"].fillna("None")

train["BsmtQual"] = train["BsmtQual"].fillna("None")

train["BsmtFinType2"] = train["BsmtFinType2"].fillna("None")

train["BsmtFinType1"] = train["BsmtFinType1"].fillna("None")



train["GarageYrBlt"] = train["GarageYrBlt"].fillna(0)

train["GarageArea"] = train["GarageArea"].fillna(0)

train["GarageCars"] = train["GarageCars"].fillna(0)

train["BsmtHalfBath"] = train["BsmtHalfBath"].fillna(0)

train["BsmtFullBath"] = train["BsmtFullBath"].fillna(0)

train["TotalBsmtSF"] = train["TotalBsmtSF"].fillna(0)

train["BsmtUnfSF"] = train["BsmtUnfSF"].fillna(0)

train["BsmtFinSF2"] = train["BsmtFinSF2"].fillna(0)

train["BsmtFinSF1"] = train["BsmtFinSF1"].fillna(0)

train["MasVnrArea"] = train["MasVnrArea"].fillna(0)



train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])

train["Functional"] = train["Functional"].fillna(train['Functional'].mode()[0])

train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])

train['Exterior1st'] = train['Exterior1st'].fillna(train['Exterior1st'].mode()[0])

train['Exterior2nd'] = train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])

train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])



train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))





test['PoolQC'] = test['PoolQC'].fillna("None")

test['MiscFeature'] = test['MiscFeature'].fillna("None")

test["Alley"] = test["Alley"].fillna("None")

test["Fence"] = test["Fence"].fillna("None")

test["FireplaceQu"] = test["FireplaceQu"].fillna("None")

test["MasVnrType"] = test["MasVnrType"].fillna("None")

test["GarageType"] = test["GarageType"].fillna("None")

test["GarageFinish"] = test["GarageFinish"].fillna("None")

test["GarageQual"] = test["GarageQual"].fillna("None")

test["GarageCond"] = test["GarageCond"].fillna("None")

test["BsmtCond"] = test["BsmtCond"].fillna("None")

test["BsmtExposure"] = test["BsmtExposure"].fillna("None")

test["BsmtQual"] = test["BsmtQual"].fillna("None")

test["BsmtFinType2"] = test["BsmtFinType2"].fillna("None")

test["BsmtFinType1"] = test["BsmtFinType1"].fillna("None")



test["GarageYrBlt"] = test["GarageYrBlt"].fillna(0)

test["GarageArea"] = test["GarageArea"].fillna(0)

test["GarageCars"] = test["GarageCars"].fillna(0)

test["BsmtHalfBath"] = test["BsmtHalfBath"].fillna(0)

test["BsmtFullBath"] = test["BsmtFullBath"].fillna(0)

test["TotalBsmtSF"] = test["TotalBsmtSF"].fillna(0)

test["BsmtUnfSF"] = test["BsmtUnfSF"].fillna(0)

test["BsmtFinSF2"] = test["BsmtFinSF2"].fillna(0)

test["BsmtFinSF1"] = test["BsmtFinSF1"].fillna(0)

test["MasVnrArea"] = test["MasVnrArea"].fillna(0)



test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])

test["Functional"] = test["Functional"].fillna(test['Functional'].mode()[0])

test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])

test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])

test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])



test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
missing_data(allhouses)
#%%aykırı değer

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

allhouses = allhouses.drop(['Utilities','GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)

train.drop(['Utilities','GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)

test.drop(['Utilities','GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)



allhouses = pd.concat([train,test], sort=False)
allhouses['MSSubClass']=allhouses['MSSubClass'].astype(str)



allhouses = pd.get_dummies(allhouses)



train = allhouses[:len_train]

test = allhouses[len_train:]



print(train.shape)

print(test.shape)
train['SalePrice']=np.log(train['SalePrice'])

allhouses = pd.concat([train,test], sort=False)

allhouses = pd.get_dummies(allhouses)



train = allhouses[:len_train]

test = allhouses[len_train:]



train.drop(0,axis=0,inplace = True)

train.drop(1,axis=0,inplace = True)
y_train = train.SalePrice.values

y_test = train['SalePrice']
model_rd = Ridge(alpha = 4.84)

model_rd.fit(train,y_train)

y_pred_rd = model_rd.predict(train)

score_rd = np.sqrt(mean_squared_error(y_train, y_pred_rd))

print("Ridge Score :",score_rd)
model_rf = RandomForestRegressor(n_estimators = 12,max_depth = 3,n_jobs = -1)

model_rf.fit(train,y_train)

y_pred_rf = model_rf.predict(train)

score_rf = np.sqrt(mean_squared_error(y_train, y_pred_rf))

print("RandomForest Score :",score_rf)
model_gb = GradientBoostingRegressor(n_estimators = 40,max_depth = 2)

model_gb.fit(train,y_train)

y_pred_gb = model_gb.predict(train)

score_gb = np.sqrt(mean_squared_error(y_train, y_pred_gb))

print("GradientBoosting Score :",score_gb)