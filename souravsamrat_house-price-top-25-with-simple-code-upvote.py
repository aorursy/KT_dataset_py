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
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.info()
train.columns
train_c =train[['MSZoning','Street','Alley','LotShape','LandContour',
               'Utilities','LotConfig','LandSlope', 'Neighborhood', 
               'Condition1', 'Condition2', 'BldgType','HouseStyle',
               'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
               'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 
               'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1',
               'BsmtFinType2', 'Heating','HeatingQC', 'CentralAir', 
               'Electrical','KitchenQual','Functional','FireplaceQu',
               'GarageType','GarageFinish','GarageQual','GarageCond',
               'PavedDrive','PoolQC','Fence','MiscFeature','SaleType',
                'SaleCondition','MSSubClass','OverallQual','OverallCond',
                'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars',
                'MiscVal','MoSold','YrSold']]
train_c.head()
test_c =test[['MSZoning','Street','Alley','LotShape','LandContour',
               'Utilities','LotConfig','LandSlope', 'Neighborhood', 
               'Condition1', 'Condition2', 'BldgType','HouseStyle',
               'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
               'MasVnrType','ExterQual', 'ExterCond', 'Foundation', 
               'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1',
               'BsmtFinType2', 'Heating','HeatingQC', 'CentralAir', 
               'Electrical','KitchenQual','Functional','FireplaceQu',
               'GarageType','GarageFinish','GarageQual','GarageCond',
               'PavedDrive','PoolQC','Fence','MiscFeature','SaleType',
                'SaleCondition','MSSubClass','OverallQual','OverallCond',
                'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars',
                'MiscVal','MoSold','YrSold']]
frames = [train_c,test_c]
df_c = pd.concat(frames, keys=['TRAIN_C', 'TEST_C'])

df_c
df_c.isnull().sum().sort_values(ascending=False)
df_c.shape
df_c.drop(['PoolQC','MiscFeature','Fence','Alley','FireplaceQu'], axis = 1, inplace = True)
df_c.shape
for column in ['GarageCond','GarageQual','GarageFinish','GarageType','BsmtCond','BsmtExposure','BsmtQual',
               'BsmtFinType2','BsmtFinType1','MasVnrType','MSZoning','Functional','Utilities','Electrical',
               'KitchenQual','SaleType','Exterior2nd','Exterior1st','BsmtFullBath','BsmtHalfBath','GarageCars']:
    df_c[column].fillna(method='ffill',inplace=True)
df_c.isnull().sum().sort_values(ascending=False)
df_c
df_c = pd.get_dummies(df_c)
df_c.shape
train_categorical = df_c.xs('TRAIN_C')
test_categorical = df_c.xs('TEST_C')
train_n =train[['LotFrontage','LotArea','YearBuilt',
                      'YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                      '1stFlrSF', '2ndFlrSF','LowQualFinSF','GrLivArea','GarageYrBlt',
                      'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
                      '3SsnPorch','ScreenPorch','PoolArea','SalePrice']]
train_n.shape
train_n.head()
train_n.isnull().sum().sort_values(ascending=False)
train_n['GarageYrBlt'].fillna(method='ffill',inplace = True)
train_n.loc[train_n['LotFrontage'].isna(),'LotFrontage'] = float(train_n['LotFrontage'].median())
train_n.loc[train_n['MasVnrArea'].isna(),'MasVnrArea'] = float(train_n['MasVnrArea'].median())
train_n.shape
train_categorical.shape
train_final = pd.concat([train_categorical,train_n],axis = 1)
train_final.shape
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
%matplotlib inline
sns.relplot(x="LotArea",y="SalePrice",hue ="SalePrice", data = train_final)
train_final = train_final[(train_final['LotArea']<100000)]
print(train_final.shape)
sns.relplot(x="LotFrontage",y="SalePrice",hue ="SalePrice", data = train_final)
train_final = train_final[(train_final['LotFrontage']<250)]
print(train_final.shape)
sns.relplot(x="GrLivArea",y="SalePrice",hue ="SalePrice", data = train_final)
train_final = train_final[train_final['GrLivArea']<4000]
print(train_final.shape)
sns.relplot(x="MasVnrArea",y="SalePrice",hue ="SalePrice", data = train_final)
train_final = train_final[train_final['MasVnrArea']<1250]
print(train_final.shape)
sns.relplot(x="MSSubClass",y="SalePrice",hue ="SalePrice", data = train_final)
sns.distplot(train_final['SalePrice'])
train_final['SalePrice'] = np.log1p(train_final['SalePrice'])
sns.distplot(train_final['SalePrice'])
test_n =test[['LotFrontage','LotArea','YearBuilt',
                      'YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
                      '1stFlrSF', '2ndFlrSF','LowQualFinSF','GrLivArea','GarageYrBlt',
                      'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
                      '3SsnPorch','ScreenPorch','PoolArea']]
test_n.head()
test_n.isnull().sum().sort_values(ascending=False)
test_n.loc[test_n['LotFrontage'].isna(),'LotFrontage'] = float(test_n['LotFrontage'].median())
test_n.loc[test_n['MasVnrArea'].isna(),'MasVnrArea'] = float(test_n['MasVnrArea'].median())
for column in ['GarageYrBlt','TotalBsmtSF','GarageArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF']:
    test_n[column].fillna(method='ffill',inplace=True)
test_n.isnull().sum().sort_values(ascending=False)
test_final =pd.concat([test_categorical,test_n],axis = 1)
test_final.shape
train_final.shape
X_train= train_final.drop('SalePrice',axis=1).values
Y_train = train_final['SalePrice'].values
X_test = test_final.values
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso,ElasticNet,Ridge
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor,VotingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
classifier=xgb.XGBClassifier()
#params={
 #"learning_rate"    : [0.05, 0.09, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 #"n_estimators"     : [128,256,512,1024,2048,4096],
 #"max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 #"min_child_weight" : [ 1, 3, 5, 7 ],
 #"gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#"subsample"         : [0.6,0.8,0.65,0.7,0.75],
#"seed"              : [20,22,24,26,27,28],
#"reg_alpha"         : [0.00001,0.00005,0.00006,0.00007],
 #"colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
#}
#random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

#random_search.fit(X_train,Y_train)

#random_search.best_estimator_
#random_search.best_params_
xg = xgb.XGBRegressor(learning_rate=0.01,n_estimators=2048,
                                     max_depth=5, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
xg = xg.fit(X_train,Y_train)
train_pred = xg.predict(X_train)
pred = xg.predict(X_test)
xg.score(X_train,Y_train)
xg.score(X_test,pred)
import math
from sklearn.metrics import mean_squared_error
import sklearn.metrics as sklm
#lasso_mod=Lasso(alpha=0.0009,max_iter = 100)
#lasso_mod.fit(X_train,Y_train)
#y_lasso_train=lasso_mod.predict(X_train)
#y_lasso_test=lasso_mod.predict(X_test)
#math.sqrt(sklm.mean_squared_error(Y_train, y_lasso_train))
#lasso_mod.score(X_train,Y_train)
#ls = Lasso(alpha=10,normalize=True)
#ls.fit(X_train,Y_train)
#train_pred = ls.predict(X_train)
#pred = ls.predict(X_test)
#ls.score(X_train,Y_train)
#logreg = LogisticRegression()

#logreg.fit(X_train,Y_train)

#Y_pred = logreg.predict(X_test)

#logreg.score(X_train,Y_train)
#ridge = Ridge(alpha=0.1,normalize=True)
#ridge.fit(X_train,Y_train)
#train_pred = ridge.predict(X_train)
#test_pred = ridge.predict(X_test)
#ridge.score(X_train,Y_train)
#es = ElasticNet(alpha=0.001,l1_ratio=0.2)
#es.fit(X_train,Y_train)
#train_pred = es.predict(X_train)
#test_pred = es.predict(X_test)
#es.score(X_train,Y_train)
#lg = lgb.LGBMRegressor(objective='regression', 
                                       #num_leaves=4,
                                       #learning_rate=0.1, 
                                       #n_estimators=5000,
                                       #max_bin=200, 
                                       #bagging_fraction=0.75,
                                       #bagging_freq=5, 
                                       #bagging_seed=7,
                                       #feature_fraction=0.2,
                                       #feature_fraction_seed=7,
                                       #verbose=-1)
#lg = lg.fit(X_train,Y_train)
#train_pred = lg.predict(X_train)
#pred = lg.predict(X_test)
#lg.score(X_train,Y_train)
test_df11 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
submission_df = pd.DataFrame(columns=['Id', 'SalePrice'])
submission_df['Id'] = test_df11['Id']
submission_df['SalePrice'] = pred*10000
submission_df.to_csv('submission.csv', header=True, index=False)
submission_df.head(10)
submission_df.to_csv("submission.csv", index=False)
