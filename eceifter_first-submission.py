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
import seaborn as sns 
%matplotlib inline
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
#importing test and train data
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_ID = df_train['Id']
test_ID = df_test['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
df_train.drop("Id", axis = 1, inplace = True)
df_test.drop("Id", axis = 1, inplace = True)
df_test.head()
df_train.head()
print(df_train.shape)
print(df_test.shape)
train_cols = df_train.columns
test_cols = df_test.columns

train_not_test = train_cols.difference(test_cols)
train_not_test

full_data.info()
df_train[['GrLivArea','SalePrice', 'YearBuilt', 'PoolArea','GarageArea','GarageCars','KitchenAbvGr','BedroomAbvGr', 'TotRmsAbvGrd']].describe()
sns.scatterplot(x=df_train['SalePrice'], y=df_train['GrLivArea'])
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<500000)].index)
sns.scatterplot(x=df_train['SalePrice'], y=df_train['GrLivArea'])
sns.distplot(df_train['SalePrice'] , fit=norm)
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()
df_train['SalePrice'] = np.log(df_train['SalePrice']) #log

sns.distplot(df_train['SalePrice'] , fit=norm)
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
plt.show()
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train.SalePrice.values
full_data = pd.concat((df_train, df_test)).reset_index(drop=True)
full_data.drop(['SalePrice'], axis=1, inplace=True)
full_data.shape
figure = plt.figure(figsize=(12,6))
sns.heatmap(full_data.isnull(),yticklabels='')
missingdata = full_data.isnull().sum()

missingdata = missingdata.drop(missingdata[missingdata == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'missing_count':missingdata})

missing_data
print(full_data['GarageYrBlt'].mean())
print(full_data['GarageYrBlt'].median())
print(full_data['Electrical'].mode())
print(full_data['MSZoning'].mode())
full_data['Utilities'].value_counts().plot(kind='bar',figsize=[10,3])
full_data['Utilities'].value_counts() 
full_data["PoolQC"] = full_data["PoolQC"].fillna("None")
full_data["MiscFeature"] = full_data["MiscFeature"].fillna("None")
full_data["Alley"] = full_data["Alley"].fillna("None")
full_data["Fence"] = full_data["Fence"].fillna("None")
full_data["FireplaceQu"] = full_data["FireplaceQu"].fillna("None")
full_data['GarageYrBlt'] = full_data['GarageYrBlt'].fillna(full_data['GarageYrBlt'].mean())
full_data['Electrical'] = full_data['Electrical'].fillna(full_data['Electrical'].mode()[0]) # most common used SBrkr 
full_data['MSZoning'] = full_data['MSZoning'].fillna(full_data['MSZoning'].mode()[0]) #most common used RL
full_data["MasVnrType"] = full_data["MasVnrType"].fillna("None")
full_data["RoofMatl"] = full_data["RoofMatl"].fillna("None")
full_data["Exterior1st"] = full_data["Exterior1st"].fillna(full_data['Exterior1st'].mode()[0])
full_data["Exterior2nd"] = full_data["Exterior2nd"].fillna(full_data['Exterior2nd'].mode()[0])
full_data["Fireplaces"] = full_data["Fireplaces"].fillna(0)
full_data["MasVnrArea"] = full_data["MasVnrArea"].fillna(0)
full_data["TotRmsAbvGrd"] = full_data["TotRmsAbvGrd"].fillna("None")
full_data["Functional"] = full_data["Functional"].fillna("Typ")
full_data["SaleType"] = full_data["SaleType"].fillna(full_data['SaleType'].mode()[0])
full_data["GarageArea"] = full_data["GarageArea"].fillna(0)
full_data["GarageCars"] = full_data["GarageCars"].fillna(0)
full_data['KitchenQual'] = full_data['KitchenQual'].fillna(full_data['KitchenQual'].mode()[0])
full_data["LotFrontage"] = full_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
full_data = full_data.drop(['Utilities'], axis=1) #because there are two types of utilities and no need such a column

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            full_data[col] =full_data[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            full_data[col] = full_data[col].fillna('None')

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
            full_data[col] = full_data[col].fillna(0)
missingdata = full_data.isnull().sum()

missingdata = missingdata.drop(missingdata[missingdata == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'missing_count':missingdata})

missing_data
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


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(full_data[c].values)) 
    full_data[c] = lbl.transform(list(full_data[c].values))

# shape        
print('Shape full_data: {}'.format(full_data.shape))
full_data = pd.get_dummies(full_data)
print(full_data.shape)
train = full_data[:ntrain]
test = full_data[ntrain:]
test.head()
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("Lasso RMSE:", (score.mean()))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)
print("GBoost score:", (score.mean()))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(model_lgb)
print("LGB score:", (score.mean()))
y_lr = df_train.SalePrice
X_lr = train
X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(
                          X_lr, y_lr, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_lr_train, y_lr_train)
print ("R^2 is: \n", model.score(X_lr_test, y_lr_test))
predictions = model.predict(X_lr_test)
print ('RMSE is: \n', mean_squared_error(y_lr_test, predictions))
feats = test.select_dtypes(
        include=[np.number]).interpolate()

predictions = model.predict(feats)

final_predictions = np.exp(predictions)
print ('RMSE is: \n', mean_squared_error(y_lr_test,final_predictions))
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
lasso.fit(train, y_train)
lasso_train_pred = lasso.predict(train)
lasso_pred = np.expm1(lasso.predict(test.values))
print(rmsle(y_train, lasso_train_pred))
GBoost.fit(train, y_train)
GBoost_train_pred = GBoost.predict(train)
GBoost_pred = np.expm1(GBoost.predict(test.values))
print(rmsle(y_train, GBoost_train_pred))
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = GBoost_pred
sub.to_csv('submission.csv',index=False)