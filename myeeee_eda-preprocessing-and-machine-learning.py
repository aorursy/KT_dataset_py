import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import log_loss

from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
# 最大90列まで表示する
# display setting for pandas.DataFrame
pd.set_option('display.max_columns', 90)
print(df_train.shape)
df_train.head()
print(df_test.shape)
df_test.head()
print(df_submission.shape)
df_submission.head()
# ノイズの削除
# Remove outliers

df_train.plot.scatter(x='GrLivArea', y='SalePrice')
plt.show()

df_train.drop(df_train.query('GrLivArea > 4000 and SalePrice < 300000').index, axis=0, inplace=True)
# Histogram and Q-Q plot

sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
# 対数変換
# Logarithmic transformation

df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
# NaNのカウント
# NaNの割合が80%以上であれば、dropリストに追加する

# Missing value counts
# drop column when it includes more than 80% missing values
# for training data

null_rate = [df_train[col].isnull().sum() / df_train.shape[0] for col in df_train.columns]
null_count = [df_train[col].isnull().sum() for col in df_train.columns]
null_info = pd.DataFrame({'null_rate': null_rate, 'null_count': null_count})
null_info.index = df_train.columns

null_info = null_info[null_info['null_rate'] > 0.0]
null_info.sort_values('null_rate', ascending=False, inplace=True)

drop_col = list(null_info[null_info['null_rate'] > 0.8].index)
print('Dropped features: ', drop_col)

sns.barplot(x=null_info['null_rate'], y=null_info.index, orient='h')

null_info.T
# NaNのカウント
# NaNの割合が80%以上であれば、dropリストに追加する

# Missing value counts
# drop column when it includes more than 80% missing values
# for test data

null_rate = [df_test[col].isnull().sum() / df_test.shape[0] for col in df_test.columns]
null_count = [df_test[col].isnull().sum() for col in df_test.columns]
null_info = pd.DataFrame({'null_rate': null_rate, 'null_count': null_count})
null_info.index = df_test.columns

null_info = null_info[null_info['null_rate'] > 0.0]
null_info.sort_values('null_rate', ascending=False, inplace=True)

drop_col.extend(list(null_info[null_info['null_rate'] > 0.8].index))
drop_col = list(set(drop_col))
print('Dropped features: ', drop_col)

tmp = null_info.loc[null_info['null_rate'] > 0.01, 'null_rate']
sns.barplot(x=tmp, y=tmp.index, orient='h')

null_info.T
# 出現頻度が最大の値をカウントする
# 一つの値が90%以上を占めていれば、dropリストに追加する

# Remove the high frequency value
# for training data

topval_count = [df_train[col].value_counts(dropna=False).iloc[0] for col in df_train.columns]
topval_rate = [df_train[col].value_counts(dropna=False).iloc[0] / df_train.shape[0] for col in df_train.columns]
topval_info = pd.DataFrame({'topval_rate': topval_rate, 'topval_count': topval_count})
topval_info.index = df_train.columns
topval_info.sort_values('topval_rate', ascending=False, inplace=True)

fig, ax = plt.subplots(figsize=(6, 6))
tmp = topval_info.loc[topval_info['topval_rate'] > 0.65, 'topval_rate']
sns.barplot(x=tmp, y=tmp.index, orient='h', ax=ax)

drop_col.extend(list(topval_info[topval_info['topval_rate'] > 0.9].index))
drop_col = list(set(drop_col))
print('Droppend features', drop_col)

topval_info.T
# 出現頻度が最大の値をカウントする
# 一つの値が90%以上を占めていれば、dropリストに追加する

# Remove the high frequency value
# for test data

topval_count = [df_test[col].value_counts(dropna=False).iloc[0] for col in df_test.columns]
topval_rate = [df_test[col].value_counts(dropna=False).iloc[0] / df_test.shape[0] for col in df_test.columns]
topval_info = pd.DataFrame({'topval_rate': topval_rate, 'topval_count': topval_count})

topval_info.index = df_test.columns
topval_info.sort_values('topval_rate', ascending=False, inplace=True)

fig, ax = plt.subplots(figsize=(6, 6))
tmp = topval_info.loc[topval_info['topval_rate'] > 0.65, 'topval_rate']
sns.barplot(x=tmp, y=tmp.index, orient='h', ax=ax)

drop_col.extend(list(topval_info[topval_info['topval_rate'] > 0.9].index))
drop_col = list(set(drop_col))
print('Dropped features', drop_col)

topval_info.T
# Drop columns

df_train.drop(drop_col, axis=1, inplace=True)
df_test.drop(drop_col, axis=1, inplace=True)

df_train.drop('MoSold', axis=1, inplace=True)
df_test.drop('MoSold', axis=1, inplace=True)

print('train shape:', df_train.shape, 'test shape:', df_test.shape)
# Show columns which incude missing values.
df_train.loc[:, [df_train[col].isnull().sum() > 0 for col in df_train.columns]].head(2)
# Show columns which incude missing values.
df_test.loc[:, [df_test[col].isnull().sum() > 0 for col in df_test.columns]].head(2)
# 欠損値埋め
# data leakageは無視する

# Impute missing values
# This method will cause data leakage

# combine train and test data
df_all = pd.concat([df_train, df_test], sort=True)

df_all['LotFrontage'] = df_all.groupby('Neighborhood')['LotFrontage'].transform(lambda x : x.fillna(x.median()))
df_all['MasVnrType'].fillna(df_all['MasVnrType'].mode()[0], inplace=True)
df_all['MasVnrArea'].fillna(df_all['MasVnrArea'].mode()[0], inplace=True)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_all[col].fillna('None', inplace=True)
    
for col in ('GarageType', 'GarageFinish', 'GarageQual'):
    df_all[col].fillna('None', inplace=True)
    
df_all['GarageYrBlt'].fillna(0, inplace=True)
df_all['FireplaceQu'].fillna('None', inplace=True)
df_all['GarageArea'].fillna(0, inplace=True)
df_all['GarageCars'].fillna(0, inplace=True)

for col in df_all.drop('SalePrice', axis=1).columns:
    if df_all[col].isnull().sum() > 0:
        df_all[col].fillna(df_all[col].mode()[0], inplace=True)

print('Final null-count: ', df_all.drop('SalePrice', axis=1).isnull().sum().sum())
# About 'GrLivArea' feature

sns.distplot(df_all['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_all['GrLivArea'], plot=plt)
# Logarithmic transformation

df_all['GrLivArea'] = np.log1p(df_all['GrLivArea'])

sns.distplot(df_all['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_all['GrLivArea'], plot=plt)
# About 'TotalBsmtSF' feature

sns.distplot(df_all['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_all['TotalBsmtSF'], plot=plt)
# Logarithmic transformation
# Applied when the sample has basement.

df_all['HasBsmt'] = (df_all['TotalBsmtSF'] > 0).astype(int)

df_all.loc[df_all['HasBsmt'] > 0, 'TotalBsmtSF'] = np.log1p(df_all.loc[df_all['HasBsmt'] > 0, 'TotalBsmtSF'] )

sns.distplot(df_all.loc[df_all['HasBsmt'] > 0, 'TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_all.loc[df_all['HasBsmt'] > 0, 'TotalBsmtSF'], plot=plt)
# category変数をone-hot表現に変換して、リッジ回帰を行う

# One-hot encoding
df_all_dummy = pd.get_dummies(df_all, drop_first=True)

# split data
train_X = df_all_dummy[df_all_dummy['SalePrice'].notnull()].drop('SalePrice', axis=1)
train_y = df_all_dummy.loc[df_all_dummy['SalePrice'].notnull(), 'SalePrice']
test_X = df_all_dummy[df_all_dummy['SalePrice'].isnull()].drop('SalePrice', axis=1)
print('train_X:', train_X.shape, 'train_y:', train_y.shape, 'test_X:', test_X.shape)

rs = RobustScaler()
train_X = rs.fit_transform(train_X)
test_X = rs.transform(test_X)

# Cross validation with Ridge regressor
kf = KFold(n_splits=4, random_state=1, shuffle=True)
ridge_alphas = [1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
#ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))
ridge  = RidgeCV(alphas=ridge_alphas, cv=kf)
ridge.fit(train_X, train_y)
ridge.score(train_X, train_y)

pred = ridge.predict(test_X)
pred = np.expm1(pred)

df_submission['SalePrice'] = pred
df_submission.to_csv('ridge_regression.csv', index=False)
# LabelEncoderより、category値を数値に変換する
# Label encoding

train_X = df_all[df_all['SalePrice'].notnull()].drop(['Id', 'SalePrice'], axis=1)
train_y = df_all.loc[df_all['SalePrice'].notnull(), 'SalePrice']
test_X = df_all[df_all['SalePrice'].isnull()].drop(['Id', 'SalePrice'], axis=1)

for col in train_X.loc[:, train_X.dtypes == 'object'].columns:
    le = LabelEncoder()
    train_X[col] = le.fit_transform(train_X[col])
    
    test_X[col] = le.transform(test_X[col])
    
print('train_X:', train_X.shape, 'train_y:', train_y.shape, 'test_X:', test_X.shape)
train_X.head()
# Cross validation with LightGBM regressor

lgb_param = {'objective': 'regression', 'seed': 1, 'metric': 'rmse', 'max_depth': 10}

lgb_train = lgb.Dataset(train_X, train_y)

cv_results = lgb.cv(lgb_param, lgb_train, num_boost_round=100, verbose_eval=10, nfold=4, stratified=False)

nround = len(cv_results['rmse-mean'])

plt.plot(range(nround), cv_results['rmse-mean'])
plt.show()
# LightGBM training and prediction

lgb_param = {'objective': 'regression', 'seed': 1, 'metric': 'rmse', 'max_depth': 10}

lgb_train = lgb.Dataset(train_X, train_y)

model = lgb.train(lgb_param, lgb_train, num_boost_round=50)

pred = model.predict(test_X)
pred = np.expm1(pred)

lgb.plot_importance(model, height=0.5, figsize=(8, 12))

df_submission['SalePrice'] = pred
df_submission.to_csv('lightgbm.csv', index=False)
# Cross validation with XGBoost regressor

dtrain = xgb.DMatrix(train_X, label=train_y)

params = {'objective': 'reg:squarederror', 'silent': 1, 'random_state': 1, 'eta': 0.01, 'max_depth':8}
num_round = 5000

cv_results = xgb.cv(params, dtrain, num_boost_round=num_round, nfold=4, stratified=False, early_stopping_rounds=15, verbose_eval=50)

nround = len(cv_results['train-rmse-mean'])

plt.plot(range(nround), cv_results['train-rmse-mean'])
plt.plot(range(nround), cv_results['test-rmse-mean'])
plt.grid()
plt.show()
# XGBoost training and prediction

dtrain = xgb.DMatrix(train_X, label=train_y)
dtest = xgb.DMatrix(test_X)

params = {'objective': 'reg:squarederror', 'silent': 1, 'random_state': 1, 'eta': 0.01, 'max_depth':8}
num_round = 1000

watch_list = [(dtrain, 'train')]
model = xgb.train(params, dtrain, num_boost_round=num_round, evals=watch_list, verbose_eval=50)

pred = model.predict(dtest)
pred = np.expm1(pred)

df_submission['SalePrice'] = pred
df_submission.to_csv('xgboost.csv', index=False)