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
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import linear_model

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew 
print (train.columns)

print(test.columns)

print(train.shape,test.shape)
train.describe()
train.head()
test.head()
print(train.shape)
print(test.shape)
df = pd.concat([train.drop(columns=['SalePrice']), test])
df.shape
train['SalePrice'].describe()

sns.distplot(train['SalePrice']);
from scipy import stats

from scipy.stats import norm, skew


fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(train['SalePrice'] , fit=norm);



plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.subplot(1,2,2)

res = stats.probplot(train['SalePrice'], plot=plt)

plt.suptitle('Before transformation')



train.SalePrice = np.log1p(train.SalePrice )

y_train = train.SalePrice.values

y_train_orig = train.SalePrice



fig = plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(train['SalePrice'] , fit=norm);

(mu, sigma) = norm.fit(train['SalePrice'])



plt.ylabel('Frequency')

plt.title('SalePrice distribution')

plt.subplot(1,2,2)

res = stats.probplot(train['SalePrice'], plot=plt)

plt.suptitle('After transformation')
saleprice = pd.DataFrame(train.iloc[:,-1])

(mu, sigma) = norm.fit(saleprice['SalePrice'])

sns.distplot(saleprice['SalePrice'],fit=norm)

plt.title('SalePrice distribution')

null = df.isnull().sum().sort_values(ascending=False)

train_nan = (null[null>0])

dict(train_nan)

train_nan
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',

            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"PoolQC"

           ,'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']:

    df[col] = df[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2'

           ,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF'):

    df[col] = df[col].fillna(0)
total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
train_length=1460

df = pd.get_dummies(df)

X_test=df.iloc[train_length:,:]

X_train=df.iloc[:train_length,:]

X=X_train
saleprice["SalePrice"] = np.log1p(saleprice["SalePrice"])

y=saleprice

y.head()
plt.figure(figsize=(10,10))

sns.boxplot(train['SalePrice'],train["Neighborhood"])
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1, square=True);
corr_num = 10

cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index

corr_mat_sales = np.corrcoef(train[cols_corr].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(12, 9))

hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)

plt.show()
var_num = 8

vars = cols_corr[0:var_num]



sns.set()

sns.pairplot(train[vars], size = 2.5)

plt.show();
n_folds = 5

def rmse_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)

    rmse= np.sqrt(-cross_val_score(model, X_train.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error , make_scorer

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression

import random

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

from sklearn.linear_model import LinearRegression

import xgboost as xgb

from xgboost import XGBRegressor
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.04, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =random.randint(0,int(2**16)), nthread = -1)
score = rmse_cv(model_xgb)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmse_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

model_xgb.fit(X,y)
print('Predict submission')

sample_submission.to_csv("submission.csv", index=False)
submission = pd.read_csv('submission.csv')

submission.head(10)