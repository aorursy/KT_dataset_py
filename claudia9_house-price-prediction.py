#import the packages that I want to use 
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.columns
train.head()
test.head()
# get the Id for train and test set

n = train.shape[0]

train_Id = train['Id']

test_Id = test['Id']

y_train = train['SalePrice']

train.drop('Id', axis=1,inplace=True)

test.drop('Id', axis=1, inplace=True)
# combine train and test data for processing

data = pd.concat((train,test)).reset_index(drop=True)

data.drop(['SalePrice'], axis=1, inplace=True)
data.shape
sns.set_style('whitegrid')
sns.distplot(y_train)
y_train = np.log1p(y_train)
sns.distplot(y_train)
missing = (data.isnull().sum()/ data.isnull().count()).sort_values(ascending=False)

missing = pd.DataFrame({'missing percentage':missing})
missing.head(40)
data = data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
for i in ('FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType',

         'MSSubClass','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    data[i] = data[i].fillna('None')
for j in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 

            'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):

    data[j] = data[j].fillna(0)
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])

data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])

data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
#group by neighborhood and apply the median of the neighbor

data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
data = data.drop(['Utilities'], axis=1)

data["Functional"] = data["Functional"].fillna("Typ")
missing1 = (data.isnull().sum()/ data.isnull().count()).sort_values(ascending=False)

missing1 = pd.DataFrame({'missing percentage':missing1})
missing1.head()
data = pd.get_dummies(data)
data.head(10)
#split train and test set

train = data[:n]

test = data[n:]
train.shape
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3610))
Ridge = KernelRidge(alpha=1, kernel='polynomial', degree=2, coef0=2.5)
Xgboost = xgb.XGBRegressor(colsample_bytree=0.46, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.78, n_estimators=2200,

                             reg_alpha=0.464, reg_lambda=0.857,

                             subsample=0.52, silent=1,

                             random_state =7, nthread = -1)
#cross validation to calculate the rmse

n_folds = 5
kf = KFold(n_folds, shuffle=True, random_state=3610).get_n_splits(train.values)
rmse_lasso = np.sqrt(-cross_val_score(lasso, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
print(rmse_lasso.mean())
rmse_enet = np.sqrt(-cross_val_score(ENet, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
print(rmse_enet.mean())
rmse_xgboost = np.sqrt(-cross_val_score(Xgboost, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
print(rmse_xgboost.mean())
fitted = Xgboost.fit(train, y_train)
predictions = Xgboost.predict(test)
predictions = np.exp(predictions)
sub = pd.DataFrame()

sub['Id'] = test_Id

sub['SalePrice'] = predictions

sub.to_csv('submission.csv',index=False)