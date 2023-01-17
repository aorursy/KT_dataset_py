# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import linear_model

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew 
print(train.shape)
print(test.shape)
train.head()
test.head()
train.info()
train.isnull().sum()
test.isnull().sum()
test.info()
df = pd.concat([train.drop(columns=['SalePrice']), test])
df.shape
df.head(10)
corr = train.corr()
mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr, mask=mask, vmax=0.8, vmin=0.05, annot=True)
df1 = pd.DataFrame(train.corr()['SalePrice'].sort_values(ascending=False))

df1.plot(kind='bar', figsize=(12,5), color='green')

plt.title('Correlation Graph')
fig, axes = plt.subplots(nrows= 3,ncols = 3, figsize=(20,12))



axes[0,0].scatter(train['YearBuilt'], train['SalePrice'], color='red')

axes[0,1].scatter(train['GarageYrBlt'], train['SalePrice'],color='green')

axes[0,2].scatter(train['GrLivArea'], train['SalePrice'], color='blue')

axes[1,0].scatter(train['TotRmsAbvGrd'], train['SalePrice'],color='red')

axes[1,1].scatter(train['GarageCars'], train['SalePrice'],color='green')

axes[1,2].scatter(train['GarageArea'], train['SalePrice'],color='blue')

axes[2,0].scatter(train['TotalBsmtSF'], train['SalePrice'],color='red')

axes[2,1].scatter(train['1stFlrSF'], train['SalePrice'],color='green')

axes[2,2].scatter(train['OverallQual'], train['SalePrice'],color='blue')
null = df.isnull().sum().sort_values(ascending=False)

train_nan = (null[null>0])

dict(train_nan)

train_nan
plt.figure(figsize=(15,8))

sns.barplot(train_nan,train_nan.index)

plt.title('Missing Data')
df["PoolQC"] = df["PoolQC"].fillna("None")

df["MiscFeature"] = df["MiscFeature"].fillna("None")

df["Alley"] = df["Alley"].fillna("None")

df["Fence"] = df["Fence"].fillna("None")

df["FireplaceQu"] = df["FireplaceQu"].fillna("None")
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    df[col] =df[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    df[col] = df[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df[col] = df[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    df[col] = df[col].fillna('None')
df["MasVnrType"] = df["MasVnrType"].fillna("None")

df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])

df["Functional"] = df["Functional"].fillna("Typ")

df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])

df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])

df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])

df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

df['MSSubClass'] = df['MSSubClass'].fillna("None")
saleprice = pd.DataFrame(train.iloc[:,-1])

(mu, sigma) = norm.fit(saleprice['SalePrice'])

sns.distplot(saleprice['SalePrice'],fit=norm)

plt.title('SalePrice distribution')
prob = stats.probplot(saleprice['SalePrice'], plot=plt)
saleprice["SalePrice"] = np.log1p(saleprice["SalePrice"])

y=saleprice

y.head()
(mu, sigma) = norm.fit(saleprice["SalePrice"])

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(saleprice["SalePrice"], fit=norm)

plt.title('SalePrice distribution')

plt.subplot(1, 2, 2)

quantile_plot=stats.probplot(saleprice['SalePrice'], plot=plt)
plt.figure(figsize=(10,10))

sns.boxplot(df["LotFrontage"],df["Neighborhood"])
df=df.drop(['GarageYrBlt','TotRmsAbvGrd','GarageArea','PoolQC', 'MiscFeature', 'Fence','MiscVal','PoolArea','Utilities'], axis=1)
df['TotalBath'] = df['FullBath'] + df['HalfBath']*0.5 + df['BsmtFullBath'] + df['BsmtHalfBath']*0.5

df['TotalFlrSF'] = df['1stFlrSF'] + df['2ndFlrSF']

df['BsmtFinSF'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
s = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2']

for col in s:

    df.drop([col], axis =1, inplace = True)
train_length=1460
df = pd.get_dummies(df)

X_test=df.iloc[train_length:,:]

X_train=df.iloc[:train_length,:]

X=X_train
df.head()
n_folds = 5

def rmse_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)

    rmse= np.sqrt(-cross_val_score(model, X_train.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
import random

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb
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
y_pred =np.expm1(model_xgb.predict(X_test))
sales=pd.DataFrame(y_pred,columns=['SalePrice'])

sample_submission['SalePrice']=sales['SalePrice']

sample_submission.head()
sample_submission.to_csv('predict.csv',index=False)
dp = pd.read_csv('predict.csv')

dp.head(10)