# Date analytics

import pandas as pd

import numpy as np

import random as rd



# Visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import missingno as msno



#model selection

from sklearn.model_selection import GridSearchCV,train_test_split



#model metrix

from sklearn.metrics import mean_squared_error



# ensemble

from sklearn.ensemble import IsolationForest



# machine learning algo

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb



# warning

import warnings

warnings.filterwarnings('ignore')
# Gathering data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# record Id

train_df_idx = train_df['Id']

test_df_idx = test_df['Id']



#drop Id

train_df.drop(['Id'], axis=1, inplace=True)

test_df.drop(['Id'], axis=1, inplace=True)



all_df = pd.concat((train_df,test_df)).reset_index(drop=True)
all_df.head(5)
# check the Nan data

msno.matrix(all_df)
Nan_features = all_df.isna().sum()[all_df.isna().sum() != 0].sort_values(ascending=False)

Nan_features.plot(kind = "bar", figsize=(11,9))
print(all_df.dtypes[all_df.isna().sum() != 0])
# fill in the NA or None

all_df['PoolQC'].fillna('NA', inplace=True)

all_df['MiscFeature'].fillna('None', inplace=True)

all_df['Alley'].fillna('NA', inplace=True)

all_df['Fence'].fillna('NA', inplace=True)

all_df['FireplaceQu'].fillna('NA', inplace=True)

all_df['GarageCond'].fillna('NA' , inplace=True)

all_df['GarageFinish'].fillna('NA' , inplace=True)

all_df['GarageQual'].fillna('NA' , inplace=True)

all_df['GarageType'].fillna('NA' , inplace=True)

all_df['BsmtCond'].fillna('NA' , inplace=True)

all_df['BsmtExposure'].fillna('NA' , inplace=True)

all_df['BsmtFinType1'].fillna('NA' , inplace=True)

all_df['BsmtFinType2'].fillna('NA' , inplace=True)

all_df['BsmtQual'].fillna('NA' , inplace=True)

all_df['MasVnrType'].fillna('None', inplace=True)
# fill in the 0

all_df['GarageArea'].fillna(0 , inplace=True)

all_df['GarageCars'].fillna(0 , inplace=True)

all_df['GarageYrBlt'].fillna(0 , inplace=True)

all_df['BsmtFinSF1'].fillna(0 , inplace=True)

all_df['BsmtFinSF2'].fillna(0 , inplace=True)

all_df['BsmtFullBath'].fillna(0 , inplace=True)

all_df['BsmtHalfBath'].fillna(0 , inplace=True)

all_df['BsmtUnfSF'].fillna(0 , inplace=True)
# fill in the mode

all_df['MSZoning'].fillna('RL' , inplace=True)

all_df['Utilities'].fillna('AllPub' , inplace=True)

all_df['Electrical'].fillna('SBrkr' , inplace=True)

all_df['SaleType'].fillna('WD' , inplace=True)

all_df['SaleType'].fillna('TA' , inplace=True)

all_df['Exterior1st'].fillna('VinylSd' , inplace=True)

all_df['Exterior2nd'].fillna('VinylSd' , inplace=True)

all_df['Functional'].fillna('Typ' , inplace=True)

all_df['KitchenQual'].fillna('TA' , inplace=True)

all_df['GarageCars'].fillna(all_df['GarageCars'].mode() , inplace=True)
# fill in the mean

all_df['BsmtFullBath'].fillna(all_df['BsmtFullBath'].mean() , inplace=True)

all_df['BsmtHalfBath'].fillna(all_df['BsmtHalfBath'].mean() , inplace=True)

all_df['GarageArea'].fillna(all_df['GarageArea'].mean() , inplace=True)
#fill in the median

all_df['LotFrontage'].fillna(all_df['LotFrontage'].median() , inplace=True) 

all_df['TotalBsmtSF'].fillna(all_df['TotalBsmtSF'].median() , inplace=True)

all_df['MasVnrArea'].fillna(all_df['MasVnrArea'].median() , inplace=True)

all_df['BsmtFinSF2'].fillna(all_df['BsmtFinSF2'].median() , inplace=True)

all_df['BsmtFinSF1'].fillna(all_df['BsmtFinSF1'].median() , inplace=True)
all_df.isnull().sum()[all_df.isnull().sum() != 0].sort_values(ascending=False)
# hot encoding 

all_df = pd.get_dummies(all_df)

all_df.head(5)
# Delete outliner(Isoration forest)

mtrain=train_df.shape[0]

train = all_df[:mtrain]

test =  all_df[mtrain:]



X = train.drop("SalePrice",axis=1)

y = train["SalePrice"]



ilf = IsolationForest(contamination=0.01,n_estimators=1000)

ilf.fit(train)



outliners = pd.DataFrame(ilf.predict(train))

outliners.columns = ['OutlinerResult']

outliners[outliners == -1].count()



outliner_result = pd.concat([train,outliners], axis=1)

outliner_result.shape
sns.lmplot(x='1stFlrSF', y='SalePrice', hue='OutlinerResult',data=outliner_result,size=8,fit_reg=False)
#Delete Outliner 

Outliner_Delete_df = outliner_result[outliner_result['OutlinerResult'] ==1 ]

# Delete OutlinerResult

Outliner_Delete_df = Outliner_Delete_df.drop('OutlinerResult',axis=1)
#feature enginerring 



import datetime

This_Year = datetime.date.today().year



#create new feature. spendtimefromBuild: 2019 - YearBuild

Outliner_Delete_df['SpendTimeBuild'] = This_Year - Outliner_Delete_df['YearBuilt']



#GarageYrBlt: Year garage was built -> create new feature spendTimeYrBltgarage: 2019 - GarageYrBlt

Outliner_Delete_df['SpndTimeGarage'] = This_Year - Outliner_Delete_df['GarageYrBlt']



Outliner_Delete_df['SpndTimeSold']  =  This_Year - Outliner_Delete_df['YrSold']



Outliner_Delete_df['All_area_Flr'] = Outliner_Delete_df['1stFlrSF'] + Outliner_Delete_df['2ndFlrSF'] + Outliner_Delete_df['TotalBsmtSF']



Outliner_Delete_df['Equipment_Feets'] = Outliner_Delete_df['WoodDeckSF'] + Outliner_Delete_df['OpenPorchSF'] + Outliner_Delete_df['EnclosedPorch']+Outliner_Delete_df['3SsnPorch']+Outliner_Delete_df['ScreenPorch'] + Outliner_Delete_df['PoolArea']  + Outliner_Delete_df['LotArea']   
X = Outliner_Delete_df.drop("SalePrice",axis=1)

y = Outliner_Delete_df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
# machine learning algo

lasso = Lasso()

rf = RandomForestRegressor()

svr = SVR()

mod = xgb.XGBRegressor()



#Grid Search

lasso_params = {'alpha':[0.1,0.5,1]}

rf_params = {'max_depth':[4,8,12],'n_estimators':[10,100,500,1000]} 

svr_params = {'C':[1e-1,1e+1,1e+3],'epsilon':[0.05,0.1,0.3]}

xgb_params = {'n_estimators':[10,100,500,1000],'learning_rate':[0.01,0.1,0.5,1]}
#lasso

lasso_gs = GridSearchCV(lasso,lasso_params)

lasso_gs.fit(X_train,y_train)
lasso_gs.best_params_
# random forest

rf_gs = GridSearchCV(rf,rf_params)

rf_gs.fit(X_train,y_train)
rf_gs.best_params_
#svr

svr_gs = GridSearchCV(svr,svr_params)

svr_gs.fit(X_train,y_train)
svr_gs.best_params_
#xgb

mod_gs = GridSearchCV(mod,xgb_params)

mod_gs.fit(X_train,y_train)
mod_gs.best_params_
lasso_pred = lasso_gs.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,lasso_pred)))
rf_pred = rf_gs.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,rf_pred)))
svr_pred = svr_gs.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,svr_pred)))
mod_pred = mod_gs.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,mod_pred)))
reg = xgb.XGBRegressor(**mod_gs.best_params_)

reg.fit(X_train,y_train)
importances = pd.Series(reg.feature_importances_, index = X_train.columns)

importances = importances.sort_values(ascending=False)

importances_show = importances[:30]

importances_show.plot(kind = "bar", figsize=(11,9))
submit = pd.concat((test_df_idx,pd.DataFrame(mod_pred)),axis=1)

submit.columns = ['Id','SalePrice']

submit.to_csv('submission.csv', index=False)