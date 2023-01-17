import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math

pd.set_option('display.max_columns', None)

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv',index_col = 'Id')

test = pd.read_csv('../input/test.csv',index_col = 'Id')
train.info()
train.head()
train.price= np.log1p(train["price"])#mengubah skala price untuk mempermudah modelling
numerical_cols = [cname for cname in train.columns if 

                train[cname].dtype in ['int64', 'float64']]

plt.style.use('seaborn-whitegrid')

fig, axarr = plt.subplots(7, 3, figsize=(12, 21))

for i in range(len(numerical_cols)):

    sns.distplot(train[~train[numerical_cols[i]].isnull()][numerical_cols[i]],ax=axarr[int(i/3)][i%3])

fig, axarr = plt.subplots(7, 3, figsize=(12, 21))

for i in range(len(numerical_cols)):

    sns.regplot(x=numerical_cols[i],y='price',data=train,ax=axarr[int(i/3)][i%3])

train = train[~((train.bathrooms>6)& (train.price<14))]
train_corr = train.copy()

corrmat = train_corr.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(25,25))

#plot heat map

g=sns.heatmap(train_corr[top_corr_features].corr(),annot=True,cmap="RdYlGn")

g
corrmat = train_corr.corr()

top_corr_features = corrmat.index[abs(corrmat["price"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(train_corr[top_corr_features].corr(),annot=True,cmap="RdYlGn")
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.price.values

all_data = pd.concat((train, test))

all_data.drop(['price'], axis=1, inplace=True)

all_data.shape

#train dan test disatukan sementara untuk diolah lebih lanjut
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head(20)
#mengubah kolom zipcode menjadi categorical

all_data['zipcode'] = all_data['zipcode'].apply(str)
#membuat kolom baru dari kolom date

all_data['YrSold'] =all_data['date'].apply(lambda s:int(s[:4]))

all_data['MonthSold'] =all_data['date'].apply(lambda s:(s[4:6]))

all_data['DaySold'] =all_data['date'].apply(lambda s:int(s[6:8]))

all_data.drop(['date'], axis=1, inplace=True)

#membuat kolom jumlah ruangan

all_data['rooms'] = all_data['bedrooms']+all_data['bathrooms']

all_data.head()
from scipy.stats import skew

numerical_cols = [cname for cname in all_data.columns if 

                all_data[cname].dtype in ['int64', 'float64']]

skewness = all_data[numerical_cols].apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.9]

print(str(skewness.shape[0]) + " skewed numerical features to log transform")

skewed_features = list(skewness.index)

all_data[skewed_features] = np.log1p(all_data[skewed_features])
Dummies_all_data = pd.get_dummies(all_data)
#data dipisah kembali jadi train sama test

X_train = Dummies_all_data[:ntrain]

X_test = Dummies_all_data[ntrain:]
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

import lightgbm as lgb

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor



n_folds=5

def rmsle_cv(model):

    rmsle= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))

    print("\nscore rmsle: {:.4f} ({:.4f})\n".format(rmsle.mean(), rmsle.std()))

    return(rmsle)

model_xgb = XGBRegressor(colsample_bytree=1, gamma=0.0468, 

                             learning_rate=0.053, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                              subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

score_xgb = rmsle_cv(model_xgb)

#0.1878 (0.0041)
model_LGB = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score_LGB = rmsle_cv(model_LGB)
model_GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

score_GBoost = rmsle_cv(model_GBoost)
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

score_ENet = rmsle_cv(ENet)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score_lasso = rmsle_cv(lasso)
model_GBoost.fit(X_train,y_train)

model_xgb.fit(X_train,y_train)

model_LGB.fit(X_train,y_train)

lasso.fit(X_train,y_train)

ENet.fit(X_train,y_train)

lasso.fit(X_train,y_train)

preds_test = np.expm1( model_xgb.predict(X_test)*0.50 + (

    model_LGB.predict(X_test)*0.10+ model_GBoost.predict(X_test)*0.30) + (

lasso.predict(X_test)*0.05+ENet.predict(X_test)*0.05))

#bobot didapat berdasarkan nilai cross validation tiap model (coba - coba)

output = pd.DataFrame({'Id': X_test.index,

                       'price': preds_test})

output.to_csv('submission.csv', index=False)
output