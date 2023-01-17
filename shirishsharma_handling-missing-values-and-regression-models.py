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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

%matplotlib inline



from scipy import stats



from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge

from sklearn.model_selection import train_test_split,RandomizedSearchCV

from sklearn.ensemble import GradientBoostingRegressor,VotingRegressor

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
train = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train.info()
test.info()
train.describe()
test.describe()
train.isnull().any().describe()
test.isnull().any().describe()
print(train.shape)

print(test.shape)
Y = train['SalePrice']

train = train.drop('SalePrice',axis=1)

data = pd.concat([train,test],axis=0)

data = data.reset_index(drop=True)

data.shape
print("Number of duplicate values in train set : ",train.duplicated().sum())

print("Number of duplicate values in test set : ",test.duplicated().sum())
data_null = (data.isnull().sum() / len(data)) * 100

print(data_null)

data_null = data_null.drop(data_null[data_null == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :data_null})

ms = (missing_data.head(30)).style.background_gradient(low=0,high=1,axis=0,cmap='Oranges')

ms
data['MSZoning'].unique()
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
data['LotFrontage'].unique()
data['LotFrontage'].median()
data['LotFrontage'] = data['LotFrontage'].fillna(68)
data['Alley'].unique()
data["Alley"] = data["Alley"].fillna("NA")
data['Utilities'].unique()
data['Utilities'].isnull().sum()
data['Utilities'] = data['Utilities'].fillna('AllPub')
print(data['Exterior1st'].unique())

print(data['Exterior2nd'].unique())
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
print(data['MasVnrType'].unique())

print(data['MasVnrArea'].unique())
data["MasVnrType"] = data["MasVnrType"].fillna("NA")

data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
print(data['BsmtQual'].unique())

print(data['BsmtCond'].unique())

print(data['BsmtExposure'].unique())

print(data['BsmtFinType1'].unique())

print(data['BsmtFinType2'].unique())
data['BsmtQual'] = data['BsmtQual'].fillna('NA')

data['BsmtCond'] = data['BsmtCond'].fillna('NA')

data['BsmtFinType1'] = data['BsmtFinType1'].fillna('NA')

data['BsmtExposure'] = data['BsmtExposure'].fillna('NA')

data['BsmtFinType2'] = data['BsmtFinType2'].fillna('NA')
data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0)

data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0)

data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(0)

data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0)

data['BsmtFullBath'] = data['BsmtFullBath'].fillna(0)

data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(0)
data['KitchenQual'].unique()
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
data['Electrical'].unique()
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
data['Functional'].unique()
data['Functional'] = data['Functional'].fillna(data['Functional'].mode()[0])
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)

data['GarageArea'] = data['GarageArea'].fillna(0)

data['GarageCars'] = data['GarageCars'].fillna(0)
data['GarageType'] = data['GarageType'].fillna('NA')

data['GarageFinish'] = data['GarageFinish'].fillna('NA')

data['GarageQual'] = data['GarageQual'].fillna('NA')

data['GarageCond'] = data['GarageCond'].fillna('NA')
data['FireplaceQu'].unique()
data["FireplaceQu"] = data["FireplaceQu"].fillna('NA')
data['Fence'].unique()
data["Fence"] = data["Fence"].fillna('NA')
data['MiscFeature'].unique()
data["MiscFeature"] = data["MiscFeature"].fillna("NA")
data['PoolQC'].unique()
data["PoolQC"] = data["PoolQC"].fillna("NA")
data['SaleType'].unique()
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data['MSSubClass'].unique()
data['MSSubClass'] = data['MSSubClass'].fillna("NA")
data_null = (data.isnull().sum() / len(data)) * 100

print(data_null)
train = data[:train.shape[0]]

test = data[train.shape[0]:]

train['SalePrice'] = Y
print(train.columns.values)

print(train.shape)

print(test.shape)
fig = px.scatter(train,x='LotArea',y='SalePrice',color='SalePrice',size='SalePrice')

fig.show()
train = train[train['LotArea']<100000]

print(train.shape)
fig = px.scatter(train,x='LotFrontage',y='SalePrice',color='SalePrice',size='SalePrice')

fig.show()
train = train.drop(train[(train['LotFrontage']>300) & (train['SalePrice']<300000)].index)
fig = px.scatter(train,x='GrLivArea',y='SalePrice',size='SalePrice',color='SalePrice')

fig.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
fig  = px.scatter(train,x='LandSlope',y='SalePrice',size='SalePrice',color='SalePrice')

fig.show()
train = train.drop(train[(train['LandSlope']=='Gtl') & (train['SalePrice']>700000)].index)

train = train.drop(train[(train['LandSlope']=='Mod') & (train['SalePrice']>500000)].index)

train = train.drop(train[(train['LandSlope']=='Sev') & (train['SalePrice']>200000)].index)
train.shape
fig = px.scatter(train,x='Heating',y='SalePrice',size='SalePrice',color='SalePrice')

fig.show()
fig = px.scatter(train,x='MSSubClass',y='SalePrice',size='SalePrice',color='SalePrice')

fig.show()
fig = px.scatter(train,x='MasVnrArea',y='SalePrice',size='SalePrice',color='SalePrice')

fig.show()
train = train.drop(train[(train['MasVnrArea']>1200)].index)
plt.figure(figsize=(20,12))

plt.subplot(2,2,1)

sns.distplot(train['SalePrice'],color='green',bins=10)

plt.grid()

plt.title("Sale Price Values distribution")



sp = np.asarray(train['SalePrice'].values)

saleprice_transformed = stats.boxcox(sp)[0]



plt.subplot(2,2,2)

sns.distplot(saleprice_transformed,color='red',bins=10)

plt.grid()

plt.title("Box-Cox transformed Sale Price Values")



plt.show()
skewed_features = pd.DataFrame(train.skew().sort_values(ascending=False))

skewed_features = skewed_features.style.background_gradient(low=0,high=1,cmap='Purples',axis=0)

skewed_features
plt.figure(figsize=(25,20))

sns.heatmap(train.corr(),cmap='Oranges',fmt=".3f",annot=True)

plt.show()
print(train.shape)

print(test.shape)
Y = train['SalePrice']

train = train.drop('SalePrice',axis=1)

data = pd.concat([train,test],axis=0)

data_ohe = pd.get_dummies(data)

train_ohe = data_ohe[:train.shape[0]]

test_ohe = data_ohe[train.shape[0]:]
print(train_ohe.shape)

print(test_ohe.shape)

print(Y.shape)
X_train,X_test,Y_train,Y_test = train_test_split(train_ohe,Y,test_size=0.2)
print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
X_train.isnull().sum()
rbscaler = RobustScaler()

X_train = rbscaler.fit_transform(X_train)

X_test = rbscaler.fit_transform(X_test)

test_ohe = rbscaler.fit_transform(test_ohe)
lr = LinearRegression()

lr.fit(X_train,Y_train)

train_pred = lr.predict(X_train)

pred = lr.predict(X_test)

print("Mean Squared Error on test data : ",mean_squared_error(Y_test,pred))

print("Mean Squared Error on train data : ",mean_squared_error(Y_train,train_pred))

rmse= np.sqrt(mean_squared_error(Y_test,pred))

rmse_train = np.sqrt(mean_squared_error(Y_train,train_pred))

print("Test rmse :",rmse)

print("Train rmse :",rmse_train)
# params = {

#     'alpha':[0.0001,0.001,0.01,0.1,0.2,0.3,0.311,0.4,1,10,100],

# }

# lasso = Lasso(normalize=True)



# clf = RandomizedSearchCV(lasso,params,n_jobs=-1,verbose=0,cv=10,scoring='neg_mean_squared_error')

# clf.fit(X_train,Y_train)



# print("Best parameters  :",clf.best_params_)
ls = Lasso(alpha=10,normalize=True)

ls.fit(X_train,Y_train)

train_pred = ls.predict(X_train)

test_pred = ls.predict(X_test)

print("Root Mean Square Error for train data is : ",np.sqrt(mean_squared_error(Y_train, train_pred)))

print("Root Mean Square Error test data is  : ",np.sqrt(mean_squared_error(Y_test, test_pred)))
# params = {

#     'alpha':[0.0001,0.001,0.01,0.1,0.2,0.3,0.311,0.4,1,10,100],

# }

# ridge = Ridge(normalize=True)



# clf = RandomizedSearchCV(ridge,params,n_jobs=-1,verbose=0,cv=10,scoring='neg_mean_squared_error')

# clf.fit(X_train,Y_train)



# print("Best parameters  :",clf.best_params_)
ridge = Ridge(alpha=0.1,normalize=True)

ridge.fit(X_train,Y_train)

train_pred = ridge.predict(X_train)

test_pred = ridge.predict(X_test)

print("Root Mean Square Error for train data is : ",np.sqrt(mean_squared_error(Y_train, train_pred)))

print("Root Mean Square Error test data is  : ",np.sqrt(mean_squared_error(Y_test, test_pred)))
# params = {

#     'alpha':[0.0001,0.001,0.01,0.1,0.2,0.3,0.311,0.4,1,10,100],

#     'l1_ratio':[0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,10]

# }

# es = ElasticNet(normalize=True)



# clf = RandomizedSearchCV(es,params,n_jobs=-1,verbose=0,cv=10,scoring='neg_mean_squared_error')

# clf.fit(X_train,Y_train)



# print("Best parameters  :",clf.best_params_)
es = ElasticNet(alpha=0.001,l1_ratio=0.2)

es.fit(X_train,Y_train)

train_pred = es.predict(X_train)

test_pred = es.predict(X_test)

print("Root Mean Square Error for train data is : ",np.sqrt(mean_squared_error(Y_train, train_pred)))

print("Root Mean Square Error test data is  : ",np.sqrt(mean_squared_error(Y_test, test_pred)))
# xg_reg = xgb.XGBRegressor()

# xgparam_grid= {'learning_rate' : [0.01],'n_estimators':[2000, 3460, 4000],

#                                     'max_depth':[3], 'min_child_weight':[3,5],

#                                     'colsample_bytree':[0.5,0.7],

#                                     'reg_alpha':[0.0001,0.001,0.01,0.1,10,100],

#                                    'reg_lambda':[1,0.01,0.8,0.001,0.0001]}



# xg_grid=RandomizedSearchCV(xg_reg,xgparam_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# xg_grid.fit(X_train,Y_train)

# print(xg_grid.best_estimator_)

# print(xg_grid.best_score_)
xg = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)

xg = xg.fit(X_train,Y_train)

train_pred = xg.predict(X_train)

pred = xg.predict(X_test)

print("Root Mean Square Error on train data is :",np.sqrt(mean_squared_error(Y_train, train_pred)))

print("Root Mean Square Error on test data is :",np.sqrt(mean_squared_error(Y_test, pred)))
# params = {

#     'learning_rate':[0.001,0.01,0.002,0.003,0.004,0.1,1,10],'n_estimators':[5,10,15,25,30,35,20,40,50,70,90,100,200,400,500,1000,1500,2000,5000],

#     'max_depth':[2,5,10,12,15,17,19,20,22,25,27,30,32,35,37,39,40,41,43,45,47,49,50,60,70,80,90,100,150,200],

#     'num_leaves' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

# }

# lg = lgb.LGBMRegressor()

# lg=RandomizedSearchCV(lg,params, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)

# lg.fit(X_train,Y_train)

# print(lg.best_estimator_)

# print(lg.best_score_)
lg = lgb.LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1)

lg = lg.fit(X_train,Y_train)

train_pred = lg.predict(X_train)

pred = lg.predict(X_test)

print("Root Mean Square Error on train data is :",np.sqrt(mean_squared_error(Y_train, train_pred)))

print("Root Mean Square Error on test data is :",np.sqrt(mean_squared_error(Y_test, pred)))
gbdt = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)

gbdt = gbdt.fit(X_train,Y_train)

train_pred = gbdt.predict(X_train)

pred = gbdt.predict(X_test)

print("Root Mean Square Error on train data is :",np.sqrt(mean_squared_error(Y_train, train_pred)))

print("Root Mean Square Error on test data is :",np.sqrt(mean_squared_error(Y_test, pred)))
vc = VotingRegressor([('LGBM',lg),('XGB',xg),('ElasticNet',es)])

vc = vc.fit(X_train,Y_train)

train_pred = vc.predict(X_train)

pred = vc.predict(X_test)

print("Root Mean Square Error on train data is :",np.sqrt(mean_squared_error(Y_train, train_pred)))

print("Root Mean Square Error on test data is :",np.sqrt(mean_squared_error(Y_test, pred)))
test = test.reset_index(drop=True)

test['Id']
submit = pd.DataFrame(test['Id'],columns=['Id'])

predictions = vc.predict(test_ohe)

submit['SalePrice'] = predictions

len(submit)
submit.to_csv("submission.csv",index=False)

print("File Saved...")