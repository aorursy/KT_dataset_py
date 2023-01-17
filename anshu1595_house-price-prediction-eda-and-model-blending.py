import warnings

warnings.filterwarnings('ignore')
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
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

import scipy.stats
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
train.head()
test.head()
train.columns
corrmat = train.corr()

plt.figure(figsize=(12,8))

sns.heatmap(corrmat, cmap=plt.cm.RdBu_r)
top_corr_cols = corrmat['SalePrice'].nlargest(11).index.tolist()

top_corrmat = train[top_corr_cols].corr()

plt.figure(figsize=(12,8))

sns.heatmap(top_corrmat, annot=True, cmap=plt.cm.RdBu_r)
top_corr_cols
train[top_corr_cols].info()
sns.pairplot(train[['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd']], 

             height=2 )
train.plot.scatter('GrLivArea','SalePrice')

train['GrLivArea'].nlargest()
train_1=train[train['GrLivArea']<4500]

train_1.shape
train_1.plot.scatter('GarageArea','SalePrice')

train_1['GarageArea'].nlargest()
train_1=train_1[train_1['GarageArea']<1230]

train_1.shape
train_1.plot.scatter('TotalBsmtSF','SalePrice')

train_1['TotalBsmtSF'].nlargest()
train_1=train_1[train_1['TotalBsmtSF']<3100]

train_1.shape
train_1.plot.scatter('1stFlrSF','SalePrice')

train_1['1stFlrSF'].nlargest()
for c in top_corr_cols:

    plt.figure(figsize=(15,5))

    plt.subplot(1,2, 1)

    sns.distplot(train_1[c], fit=norm)

    plt.subplot(1,2,2)

    _ = scipy.stats.probplot(train_1[c],plot=plt)
for c in ['SalePrice','GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF']:

    plt.figure(figsize=(25,4))

    plt.subplot(1,4, 1)

    sns.distplot(train_1[c], fit=norm)

    plt.subplot(1,4,2)

    _ = scipy.stats.probplot(train_1[c],plot=plt)

    plt.subplot(1,4, 3)

    sns.distplot(np.log1p(train_1[c]), fit=norm)

    plt.subplot(1,4,4)

    _ = scipy.stats.probplot(np.log1p(train_1[c]),plot=plt)
print('\nskew before transformation:\n')

for c in ['SalePrice','GrLivArea','1stFlrSF']:

    sk = train_1[c].skew()

    print('{} : {}'.format(c,sk))

    

print('\nskew after transformation:\n')

for c in ['SalePrice','GrLivArea','1stFlrSF']:

    sk = np.log1p(train_1[c]).skew()

    print('{} : {}'.format(c,sk))
train_1['SalePriceLog'] = np.log1p(train_1['SalePrice'])

train_1['GrLivAreaLog'] = np.log1p(train_1['GrLivArea'])

train_1['1stFlrSFLog'] = np.log1p(train_1['1stFlrSF'])



test['GrLivAreaLog'] = np.log1p(test['GrLivArea'])

test['1stFlrSFLog'] = np.log1p(test['1stFlrSF'])
add_log_cols = ['SalePriceLog','GrLivAreaLog','1stFlrSFLog']

train_1[add_log_cols].info()



test[['GrLivAreaLog','1stFlrSFLog']].info()
cat_cols = [x for x in train_1.columns if train_1[x].dtype not in ('int64','float64')]

numeric_cols = [x for x in train_1.columns if train_1[x].dtype in ('int64','float64')]
numeric_cols
for c in ['SalePriceLog','GrLivArea','1stFlrSF','SalePrice','Id']:

    numeric_cols.remove(c)
numeric_cols
cat_cols
features_train = train_1.reset_index(drop=True)

y = train_1['SalePriceLog'].reset_index(drop=True)

features_train = features_train.drop(['Id','SalePrice','GrLivArea','1stFlrSF','SalePriceLog'], axis=1)



features_test = test.drop(['Id','GrLivArea','1stFlrSF'], axis=1).reset_index(drop=True)



features_train.shape, y.shape, features_test.shape
all_data = pd.concat([features_train,features_test], ignore_index=True).reset_index(drop=True)

all_data.shape
all_data.isnull().sum().nlargest(35)
all_data.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis=1, inplace=True)
cat_cols = [x for x in all_data.columns if all_data[x].dtype not in ('int64','float64')]

numeric_cols = [x for x in all_data.columns if all_data[x].dtype in ('int64','float64')]
[x for x in all_data.columns if max(all_data[x].value_counts())>2500]
all_data['Street'].value_counts()
all_data.drop(['Street','Utilities'],axis=1,inplace=True)
cat_cols = [x for x in all_data.columns if all_data[x].dtype not in ('int64','float64')]

numeric_cols = [x for x in all_data.columns if all_data[x].dtype in ('int64','float64')]
all_data.isnull().sum().nlargest(30)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
all_data['GarageQual'].fillna('None', inplace=True)

all_data['GarageFinish'].fillna('None', inplace=True)

all_data['GarageCond'].fillna('None', inplace=True)

all_data['GarageType'].fillna('None', inplace=True)



all_data['BsmtExposure'].fillna('None', inplace=True)

all_data['BsmtCond'].fillna('None', inplace=True)

all_data['BsmtQual'].fillna('None', inplace=True)

all_data['BsmtFinType2'].fillna('None', inplace=True)

all_data['BsmtFinType1'].fillna('None', inplace=True)
all_data['GarageYrBlt'] = (all_data['YearBuilt'] + all_data['YearRemodAdd']) /2
all_data['MasVnrType'].fillna('None', inplace=True)

all_data['MasVnrArea'].fillna(all_data['MasVnrArea'].mean(), inplace=True)

all_data['HasMasVnr'] = all_data['MasVnrType'].apply(lambda x: 0 if x == 'None' else 1)



all_data['MSZoning'] = all_data.groupby(['MSSubClass'])['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
for c in ['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea']:

    all_data[c].fillna(0, inplace=True)
for c in ['Functional','Exterior1st','Exterior2nd','Electrical','KitchenQual','SaleType']:

    all_data[c].fillna((all_data[c].value_counts().index[0]),inplace=True)
all_data.isnull().sum().nlargest()
all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']



all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSFLog'] + all_data['2ndFlrSF']



all_data['TotalSqrFootage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] + all_data['1stFlrSFLog'] + all_data['2ndFlrSF'])



all_data['TotalBathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))



all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF'])
all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
all_data.shape
all_data_dum = pd.get_dummies(all_data, drop_first=True)

all_data_dum.shape
X = all_data_dum.iloc[:len(y),:]

X_test= all_data_dum.iloc[len(y):,:]

X.shape, y.shape, X_test.shape
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.pipeline import make_pipeline
def rmse_cv(model):

    neg_mse = cross_val_score(model, X,y, cv=5, scoring='neg_mean_squared_error')

    rmse=np.sqrt(-neg_mse)

    print('RMSE mean: {0:.5f} , RMSE std: {1:.5f}'.format(rmse.mean(), rmse.std()))
model_ridge = make_pipeline(RobustScaler(), Ridge(alpha=7, random_state=1))

rmse_cv(model_ridge)
model_lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.00022, random_state=1))

rmse_cv(model_lasso)
model_enet = make_pipeline(RobustScaler(), ElasticNet(l1_ratio=2.4589795918367347, alpha=0.0001, random_state=1))

rmse_cv(model_enet)
model_xgboost = XGBRegressor(base_score=0.5,gamma=0.009,learning_rate=0.09000000000000001, max_depth=3,

             min_child_weight=7,n_estimators=500, n_jobs=-1,random_state=0,

             reg_alpha=0.1811111111111111, reg_lambda=0.0788888888888889)

rmse_cv(model_xgboost)
model_rf = make_pipeline(StandardScaler(), RandomForestRegressor(n_jobs=-1))

rmse_cv(model_rf)
model_svr = make_pipeline(StandardScaler(), SVR(C=20, epsilon=0.001, gamma=0.0003))

rmse_cv(model_svr)
model_stacked = StackingCVRegressor([model_ridge,model_lasso,model_enet,model_xgboost,model_rf,model_svr],

                                    meta_regressor=model_ridge, use_features_in_secondary=True)





neg_mse = cross_val_score(model_stacked, np.array(X),np.array(y), cv=5, scoring='neg_mean_squared_error')

rmse=np.sqrt(-neg_mse)

print('RMSE mean: {0:.5f} , RMSE std: {1:.5f}'.format(rmse.mean(), rmse.std()))
from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=123)
def rmse(y, y_pred):

    print('RMSE: {0}'.format(np.sqrt(mean_squared_error(y, y_pred))))
def combine_models_pred(X):

    return((0.1 * model_ridge.predict(X)) + (0.15 * model_lasso.predict(X)) + ( 0.15* model_enet.predict(X)) + (0.15 * model_xgboost.predict(X)) +

           (0.1 * model_rf.predict(X)) + (0.15 * model_svr.predict(X)) + (0.2 * model_stacked.predict(np.array(X))))
model_ridge.fit(X_train,y_train)

model_lasso.fit(X_train,y_train)

model_enet.fit(X_train,y_train)

model_xgboost.fit(X_train,y_train)

model_rf.fit(X_train,y_train)

model_svr.fit(X_train,y_train)

model_stacked.fit(np.array(X_train),np.array(y_train))
rmse(y_valid, combine_models_pred(X_valid))
model_ridge.fit(X,y)

model_lasso.fit(X,y)

model_enet.fit(X,y)

model_xgboost.fit(X,y)

model_rf.fit(X,y)

model_svr.fit(X,y)

model_stacked.fit(np.array(X),np.array(y))
y_pred_test = combine_models_pred(X_test)
output = pd.DataFrame({'Id': test['Id'],

                       'SalePrice': np.expm1(y_pred_test)})

output.to_csv('submission.csv', index=False)