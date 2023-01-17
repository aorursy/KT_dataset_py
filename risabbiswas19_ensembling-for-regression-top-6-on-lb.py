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
#import some of the necessary libraries

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

from scipy.stats import norm

import seaborn as sns

sns.set(rc={'figure.figsize':(15,12)})

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train_df.head(10)
train_df.columns
len(train_df.columns)
train_df['SalePrice'].describe()
train_df.shape
train_ID = train_df['Id']

test_ID = test_df['Id']

train_df.drop("Id", axis = 1, inplace = True)

test_df.drop("Id", axis = 1, inplace = True)

#Deleting outliers

train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)
sns.set(rc={'figure.figsize':(18,8)})

sns.distplot(train_df['SalePrice'],fit=norm)



(mu, sig) = norm.fit(train_df['SalePrice'])

#Now plot the distribution

plt.legend(['Normal Distribution Curve ($\mu=$ {:.2f} & $\sigma=$ {:.2f} )'.format(mu, sig)])

plt.ylabel('Frequency')

plt.show()
print("Skewness of Sale Price is: ",train_df['SalePrice'].skew())
train_df['SalePrice'] = np.log(train_df['SalePrice']+1)

sns.distplot(train_df['SalePrice'],fit=norm)



(mu, sig) = norm.fit(train_df['SalePrice'])

#Now plot the distribution

plt.legend(['Normal Distribution Curve ($\mu=$ {:.2f} & $\sigma=$ {:.2f} )'.format(mu, sig)])

plt.ylabel('Frequency')

plt.show()
print("Skewness of Sale Price is: ",train_df['SalePrice'].skew())
corremap = train_df.corr()

plt.subplots(figsize=(15,12))

sns.heatmap(corremap, vmax=0.9, square=True)
sns.set()

columns = ['OverallQual', 'GrLivArea', 'TotalBsmtSF','GarageCars', 'GarageArea','1stFlrSF', 'FullBath', 'YearBuilt','SalePrice']

sns.pairplot(train_df[columns], size = 2)

plt.show();
var = 'GrLivArea'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

data.plot.scatter(var,'SalePrice');
#scatter plot grlivarea/saleprice

var = 'TotalBsmtSF'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

data.plot.scatter(var,'SalePrice');
#scatter plot grlivarea/saleprice

var = 'GarageArea'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

data.plot.scatter(var,'SalePrice');
#scatter plot grlivarea/saleprice

var = '1stFlrSF'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

data.plot.scatter(var, 'SalePrice');
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

sns.boxplot(x=var,y='SalePrice',hue=var,data=data)
#box plot overallqual/saleprice

f, ax = plt.subplots(figsize=(20, 16))

plt.xticks(rotation='90')

var = 'YearBuilt'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

sns.boxplot(x=var,y='SalePrice',data=data)
#box plot overallqual/saleprice

var = 'GarageCars'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

sns.violinplot(x=var,y='SalePrice',data=data,palette='rainbow', hue = 'GarageCars')
var = 'FullBath'

data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)

sns.violinplot(x=var,y='SalePrice',data=data,palette='rainbow', hue = 'FullBath')
train_df.shape
ntrain = train_df.shape[0]

ntest = test_df.shape[0]

y_train = train_df.SalePrice.values

comp_data = pd.concat((train_df, test_df)).reset_index(drop=True)

comp_data.drop(['SalePrice'], axis=1, inplace=True)

print("Comp_data size is : {}".format(comp_data.shape))
missing_val = comp_data.isnull().sum().sort_values(ascending=False)

missing_val_df = pd.DataFrame({'Feature':missing_val.index, 'Count':missing_val.values})

missing_val_df = missing_val_df.drop(missing_val_df[missing_val_df.Count == 0].index)

missing_val_df
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='60')

plt.title('Count of Missing Data Per Feature', fontsize=15)

sns.barplot(x = 'Feature', y = 'Count', data = missing_val_df,

            palette = 'cool', edgecolor = 'b')
comp_data["LotFrontage"] = comp_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    comp_data[col] = comp_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    comp_data[col] = comp_data[col].fillna(0)

comp_data['MSZoning'] = comp_data['MSZoning'].fillna(comp_data['MSZoning'].mode()[0])

comp_data["MasVnrArea"] = comp_data["MasVnrArea"].fillna(0)

comp_data['Electrical'] = comp_data['Electrical'].fillna(comp_data['Electrical'].mode()[0])

comp_data['SaleType'] = comp_data['SaleType'].fillna(comp_data['SaleType'].mode()[0])

comp_data['KitchenQual'] = comp_data['KitchenQual'].fillna(comp_data['KitchenQual'].mode()[0])

comp_data['Exterior1st'] = comp_data['Exterior1st'].fillna(comp_data['Exterior1st'].mode()[0])

comp_data['Exterior2nd'] = comp_data['Exterior2nd'].fillna(comp_data['Exterior2nd'].mode()[0])
for col in ('PoolQC', 'MiscFeature', 'Alley'):

    comp_data[col] = comp_data[col].fillna('None')

comp_data["MasVnrType"] = comp_data["MasVnrType"].fillna("None")

comp_data["Fence"] = comp_data["Fence"].fillna("None")

comp_data["FireplaceQu"] = comp_data["FireplaceQu"].fillna("None")

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    comp_data[col] = comp_data[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    comp_data[col] = comp_data[col].fillna('None')

comp_data['MSSubClass'] = comp_data['MSSubClass'].fillna("None")

comp_data['SaleType'] = comp_data['SaleType'].fillna(comp_data['SaleType'].mode()[0])

comp_data = comp_data.drop(['Utilities'], axis=1)

comp_data['OverallCond'] = comp_data['OverallCond'].astype(str)

comp_data["Functional"] = comp_data["Functional"].fillna("Typ")  

comp_data['MSSubClass'] = comp_data['MSSubClass'].apply(str)

comp_data['YrSold'] = comp_data['YrSold'].astype(str)

comp_data['MoSold'] = comp_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder

columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

for col in columns:

    labl = LabelEncoder() 

    labl.fit(list(comp_data[col].values)) 

    comp_data[col] = labl.transform(list(comp_data[col].values))     

print('Shape all_data: {}'.format(comp_data.shape))

comp_data['TotalSF'] = comp_data['TotalBsmtSF'] + comp_data['1stFlrSF'] + comp_data['2ndFlrSF']
comp_data.shape
from scipy.stats import norm, skew

numeric_feats = comp_data.dtypes[comp_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = comp_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    comp_data[feat] = boxcox1p(comp_data[feat], lam)
comp_data = pd.get_dummies(comp_data)

print(comp_data.shape)
train_df = comp_data[:ntrain]

test_df = comp_data[ntrain:]
train_df.shape
from sklearn.model_selection import KFold, cross_val_score, train_test_split

n_folds = 5

def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.values)

    rmse= np.sqrt(-cross_val_score(model, train_df.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Ridge,ElasticNet,Lasso

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.kernel_ridge import KernelRidge

import xgboost as xgb

import lightgbm as lgb



lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
from sklearn.metrics import mean_squared_error

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
model_xgb.fit(train_df, y_train)

xgb_train_pred = model_xgb.predict(train_df)

xgb_pred = np.expm1(model_xgb.predict(test_df))

print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train_df, y_train)

lgb_train_pred = model_lgb.predict(train_df)

lgb_pred = np.expm1(model_lgb.predict(test_df.values))

print(rmsle(y_train, lgb_train_pred))
ensemble = xgb_pred*0.5 + lgb_pred*0.5
submission = pd.DataFrame()

submission['Id'] = test_ID

submission['SalePrice'] = ensemble



print("Creating Submission File")

submission.to_csv("submission.csv", index=False)