# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Drop ID as we dont need it for now



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()

train_id = train['Id']

test_id = test['Id']

for i in [train,test]:

    i.drop('Id',inplace=True,axis=1)
#Checking and removing outliers for GrLivArea

plt.scatter(x=train['GrLivArea'],y=train['SalePrice'])

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
corrmat = train.corr()

top_f = corrmat.index[abs(corrmat['SalePrice'])>0.5]

top_f

plt.figure(figsize=(10,10))

g = sns.heatmap(train[top_f].corr(),annot=True,cmap="RdYlGn")
train.skew()
#Function to check the skewness of the data

def checkskew(col):

    sns.distplot(train[col],fit=norm)

    (mu, sigma) = norm.fit(train[col])

    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

checkskew('SalePrice')
train['SalePrice'] = np.log1p(train['SalePrice'])

checkskew('SalePrice')
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values



data = pd.concat((train,test)).reset_index(drop=True)

data.drop('SalePrice',axis=1,inplace=True)

data
data_na = (data.isnull().sum() / len(data))*100

data_na = data_na.drop(data_na[data_na==0].index).sort_values(ascending=False)



plt.xticks(rotation='90')

sns.barplot(data_na.index,data_na)

data["PoolQC"] = data["PoolQC"].fillna("None")

data["MiscFeature"] = data["MiscFeature"].fillna("None")

data["Alley"] = data["Alley"].fillna("None")

data["FireplaceQu"] = data["FireplaceQu"].fillna("None")

data["Fence"] = data["Fence"].fillna("None")

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    data[col] = data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    data[col] = data[col].fillna(0)

data['LotFrontage'] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x:x.fillna(x.median()))

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    data[col] = data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    data[col] = data[col].fillna('None')

data["MasVnrType"] = data["MasVnrType"].fillna("None")

data["MasVnrArea"] = data["MasVnrArea"].fillna(0)

data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])

data.drop('Utilities',inplace=True,axis=1)

data['Functional'] = data['Functional'].fillna('Typ')

mode_col = ['Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']

for col in mode_col:

    data[col] = data[col].fillna(data[col].mode()[0])
#Check if any more null values is present or not

data.isnull().sum().sum()
data.shape
#Now we want to convert the non-ordinal numerical data to string

data['MSSubClass'] = data['MSSubClass'].apply(str)

data['OverallCond'] = data['OverallCond'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



# process columns, apply LabelEncoder to categorical features

for i in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(data[i].values)) 

    data[i] = lbl.transform(list(data[i].values))



# shape        

print('Shape all_data: {}'.format(data.shape))
#Make one more new coloumn

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
#get all numeric data 

num = data.dtypes[data.dtypes != 'object'].index

skew_f = data[num].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewed = pd.DataFrame({'Skew':skew_f})

skewed
skewed = skewed[abs(skewed) > 0.75]

from scipy.special import boxcox1p

skewed_features = skewed.index

lam = 0.15

for feat in skewed_features:

    data[feat] = boxcox1p(data[feat], lam)
data = pd.get_dummies(data)

data.shape
train = data[:ntrain]

test = data[ntrain:]

train.shape
#First we'll make a validation fucntion

#We'll use K-Fold Cross Validation



n_folds=4

def vf(m):

    kf = KFold(n_folds,shuffle=True,random_state=12).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(m, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)

rr = KernelRidge(alpha=0.6,kernel='polynomial',degree=2,coef0=2.5)

score = vf(rr)

print("Ridge Regression Score : {:.4f}".format(score.mean()))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

score = vf(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
RidgeMd = rr.fit(train.values,y_train)

GboostMd = GBoost.fit(train.values,y_train)
finalMd = (np.expm1(RidgeMd.predict(test.values)) + np.expm1(GboostMd.predict(test.values)) ) / 2

finalMd
ans = pd.DataFrame()

ans['Id'] = test_id

ans['SalePrice'] = finalMd

ans.to_csv('submission.csv',index=False)