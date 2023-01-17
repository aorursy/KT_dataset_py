# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from scipy.special import boxcox1p
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
%matplotlib inline
sns.set_context('talk')
sns.set_style('dark')
plt.rcParams["figure.figsize"] = (20,20)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# columns where NaN values have meaning e.g. no pool etc.
cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2']

# replace 'NaN' with 'None' in these columns
for col in cols_fillna:
    train[col].fillna('None',inplace=True)
    test[col].fillna('None',inplace=True)
train.head()
train.info()
train.describe()
numeric_features = train.dtypes[train.dtypes != "object"].index.drop(['SalePrice', 'Id'])
print(numeric_features.values)
category_features = train.dtypes[train.dtypes == "object"].index
print(category_features.values)
(train.isnull().sum()/train.count() * 100).sort_values(ascending=False).head(10)
sns.distplot(train['SalePrice'])
print('Skewness: {}'.format(train['SalePrice'].skew()))
print('Kurtosis: {}'.format(train['SalePrice'].kurt()))
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'] )
print('Skewness: {}'.format(train['SalePrice'] .skew()))
print('Kurtosis: {}'.format(train['SalePrice'] .kurt()))
train[numeric_features].apply(lambda x: (x.dropna().skew())).sort_values()
sns.heatmap(train.drop(['Id', 'SalePrice'], axis=1).corr(), linewidths=.1)
sns.scatterplot(train['GrLivArea'], train['SalePrice'])
f, axes = plt.subplots(43, 1, figsize=(15, 500))
for i in range(43):
    sns.boxplot(train[train.dtypes[train.dtypes == "object"].index[i]], train['SalePrice'], ax=axes[i])
train = train.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'], axis=1)

numeric_features = numeric_features.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'])

skewed_feats = train[numeric_features].apply(lambda x: (x.dropna().skew())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.65]
skewed_feats = skewed_feats.index

train[skewed_feats] = boxcox1p(train[skewed_feats], 0.14)

train = pd.get_dummies(train)

train = train.fillna(train.mean())
X = train.drop(['Id','SalePrice'], axis=1)
y = train.SalePrice
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)
    rmse= np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
Ridge = Ridge()
gbr = GradientBoostingRegressor()
rfr = RandomForestRegressor()
score = rmsle_cv(lasso)
print("Lasso score: {:.5f} ({:.5f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet  score: {:.5f} ({:.5f})\n".format(score.mean(), score.std()))
score = rmsle_cv(Ridge)
print("Ridge score: {:.5f} ({:.5f})\n".format(score.mean(), score.std()))
score = rmsle_cv(gbr)
print("Gradient Boosting Regressor score: {:.5f} ({:.5f})\n".format(score.mean(), score.std()))
score = rmsle_cv(rfr)
print("Random Forest Regressor score: {:.5f} ({:.5f})\n".format(score.mean(), score.std()))