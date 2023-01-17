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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# to display all the columns in the dataset
pd.pandas.set_option('display.max_columns',None)
train_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# Let's check the shape of the train data
train_df.shape
# Let's check the shape of the test data
test_df.shape
# Let's check the head of the train data
train_df.head()
# Let's check the head of the test data
test_df.head()
train_ID=train_df['Id']
test_ID=test_df['Id']
# Histogram
plt.figure(figsize=(8,6))
sns.distplot(train_df['SalePrice']);
plt.figure(figsize=(12,5))
train_df.corr()['SalePrice'][:-1].sort_values().plot(kind='bar')
plt.xlabel('Columns')
plt.ylabel('Correlation')
plt.show()
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.figure(figsize=(12,6))
sns.boxplot(train_df['OverallQual'],train_df['SalePrice'])
plt.figure(figsize=(19,6))
sns.boxplot(train_df['YearBuilt'],train_df['SalePrice'])
plt.xticks(rotation=90);
plt.scatter(x = train_df['TotalBsmtSF'], y = train_df['SalePrice'])
plt.scatter(x = train_df['GarageArea'], y = train_df['SalePrice'])
plt.scatter(x = train_df['GarageCars'], y = train_df['SalePrice'])
plt.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])
plt.scatter(x = train_df['OverallQual'], y = train_df['SalePrice'])
sns.distplot(train_df['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
sns.distplot(train_df['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_df['GrLivArea'], plot=plt)
categorical_variables=[feature for feature in train_df.columns if train_df[feature].dtypes=='object']
print('Total number of numerial values: {}'.format(len(categorical_variables)))

train_df[categorical_variables].head()
numerical_variables=[feature for feature in train_df.columns if train_df[feature].dtypes!='object']

print('Total number of numerial values: {}'.format(len(numerical_variables)))


train_df[numerical_variables].head()
train=train_df.shape[0]
test=test_df.shape[0]
y_train=train_df.SalePrice.values
df=pd.concat((train_df, test_df)).reset_index(drop=True)
df.drop(['SalePrice'], axis=1, inplace=True)
df.shape
# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(),cbar=None,cmap='YlGnBu')
df=df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1)
df_numerical_null=[feature for feature in df.columns if df[feature].isnull().sum() and df[feature].dtypes!='object']

for feature in df_numerical_null:
    print(feature,round(df[feature].isnull().mean(),4))
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['MasVnrArea']=df['MasVnrArea'].fillna(0)
df['BsmtFinSF1']=df['BsmtFinSF1'].fillna(0)
df['BsmtFinSF2']=df['BsmtFinSF2'].fillna(0)
df['BsmtUnfSF']=df['BsmtUnfSF'].fillna(0)
df['TotalBsmtSF']=df['TotalBsmtSF'].fillna(0)
df['BsmtFullBath']=df['BsmtFullBath'].fillna(0)
df['BsmtHalfBath']=df['BsmtHalfBath'].fillna(0)
df['GarageYrBlt']=df['GarageYrBlt'].fillna(0)
df['GarageCars']=df['GarageCars'].fillna(0)
df['GarageArea']=df['GarageArea'].fillna(0)
df_categorical_null=[feature for feature in df.columns if df[feature].isnull().sum() and df[feature].dtypes=='object']

for feature in df_categorical_null:
    print(feature,round(df[feature].isnull().mean(),4))
df['MSZoning']=df['MSZoning'].fillna('RL')
df['Utilities']=df['Utilities'].fillna('None')
df['Exterior1st']=df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd']=df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
df['MasVnrType']=df['MasVnrType'].fillna('None')
df['BsmtQual']=df['BsmtQual'].fillna('None')
df['BsmtCond']=df['BsmtCond'].fillna('None')
df['BsmtExposure']=df['BsmtExposure'].fillna('None')
df['BsmtFinType1']=df['BsmtFinType1'].fillna('None')
df['BsmtFinType2']=df['BsmtFinType2'].fillna('None')
df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])
df['KitchenQual']=df['KitchenQual'].fillna('None')
df['Functional']=df['Functional'].fillna('Typ')
df['GarageType']=df['GarageType'].fillna('None')
df['GarageFinish']=df['GarageFinish'].fillna('None')
df['GarageQual']=df['GarageQual'].fillna('None')
df['GarageCond']=df['GarageCond'].fillna('None')
df['SaleType']=df['SaleType'].fillna(df['SaleType'].mode()[0])
# Courtesy: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

numeric_feats = df.dtypes[df.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    df[feat] += 1
    df[feat] = boxcox1p(df[feat], lam)
    
df[skewed_features] = np.log1p(df[skewed_features])
# Getting rid of the Id column
df=df.drop('Id',axis=1)
df=pd.get_dummies(df)
df.head()
train=df[:train]
test=df[test:]
test=test.drop(test.index[0])
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf=KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse=np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
Ridge=KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score=rmsle_cv(Ridge)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
Lasso=make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score=rmsle_cv(Lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
Elastic=make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score=rmsle_cv(Elastic)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost=GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lasso=Lasso.fit(train.values,y_train)
Elastic_model=Elastic.fit(train.values,y_train)
Ridge_model=Ridge.fit(train.values,y_train)
GBoost_model=GBoost.fit(train.values,y_train)
Model=(np.expm1(model_lasso.predict(test.values)) + np.expm1(Elastic_model.predict(test.values))
           + np.expm1(Ridge_model.predict(test.values)) + np.expm1(GBoost_model.predict(test.values)))/4
Model
submission = pd.DataFrame()
submission['Id']=test_ID
submission['SalePrice']=Model
submission.to_csv('submission.csv',index=False)