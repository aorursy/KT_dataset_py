# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor

from fancyimpute import KNN

from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.neural_network import MLPRegressor

import xgboost as xgb

from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#basic info on the variable to predict

df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'])
corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(10,10))

sns.heatmap(corrmat, vmax=.8, square=True)
k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

#sns.set(font_scale=1)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 

                 yticklabels=cols.values, xticklabels=cols.values)

plt.show()
df_all = pd.concat((df_train, df_test)).reset_index(drop=True)

number = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/len(df_all.index)).sort_values(ascending=False)

missing = pd.concat([number, percent], axis=1, keys=['Total', 'Percent'])

missing.head(35)
#dealing with missing data by filling na with None or 0 for the relevant features

#filling with None

for i in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

          'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):

    df_all[i] = df_all[i].fillna('None')



#filling with 0

for i in ('MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',

          'GarageArea', 'GarageCars'):

    df_all[i] = df_all[i].fillna(0)
#filling the value

df_all["LotFrontage"] = df_all.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
#remaning missing values

number = df_all.isnull().sum().sort_values(ascending=False)

percent = (df_all.isnull().sum()/len(df_all.index)).sort_values(ascending=False)

missing = pd.concat([number, percent], axis=1, keys=['Total', 'Percent'])

missing.head(10)
#filling missing values with most frequent value

for i in ('MSZoning', 'Utilities', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Electrical'):

    df_all[i] = df_all[i].fillna(df_all[i].mode()[0])
#verification

df_all.isnull().sum().sort_values(ascending=False)
#onehot encoding and split the data

df_all = pd.get_dummies(df_all)

df_train = df_all[:1460]

df_test = df_all[1460:].drop(['SalePrice'], axis=1)
scaled_price = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])

sns.boxplot(scaled_price)
var2 = 'GrLivArea'

bi_var = pd.concat([df_train['SalePrice'], df_train[var2]], axis = 1)

bi_var.plot.scatter(x=var2, y='SalePrice')
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
df_train.drop(['Id'], axis=1, inplace=True)

sub_id = df_test.pop('Id')

y_train_all = df_train.pop('SalePrice')

X_train_all = df_train.values

x_train, x_test, y_train, y_test = train_test_split(X_train_all, y_train_all, test_size=0.1, random_state=10)

X_sub = df_test.values
def get_score(prediction, lables):    

    print('R2: {}'.format(r2_score(prediction, lables)))

    print('RMSLE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))#no need for the log since we log transformed the data



# Shows scores for train and validation sets    

def train_test(estimator, x_train, y_train, x_test, y_test):

    prediction_train = estimator.predict(x_train)

    # Printing train scores

    print("train:")

    get_score(prediction_train, y_train)

    print("test:")

    prediction_test = estimator.predict(x_test)

    # Printing train scores

    get_score(prediction_test, y_test)    
GB = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',

                                               min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=10)

GB = GB.fit(x_train, y_train)

train_test(GB, x_train, y_train, x_test, y_test)
BRR = linear_model.BayesianRidge(n_iter=1000)

BRR = BRR.fit(x_train, y_train)

train_test(GB, x_train, y_train, x_test, y_test)
lasso = linear_model.LassoLars(alpha=0.0001, eps=0.9, max_iter=1e6)

lasso = lasso.fit(x_train, y_train)

train_test(lasso, x_train, y_train, x_test, y_test)
elnet = linear_model.ElasticNet(alpha=0.0004, max_iter=1e5, random_state=10)

elnet = elnet.fit(x_train, y_train)

train_test(elnet, x_train, y_train, x_test, y_test)
rid = linear_model.Ridge(alpha=0.9, max_iter=1e6, random_state=10)

rid = rid.fit(x_train, y_train)

train_test(rid, x_train, y_train, x_test, y_test)


lgbm = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000, max_bin=200, bagging_fraction=0.75,

                     bagging_freq=5, bagging_seed=7, feature_fraction=0.2, feature_fraction_seed=7, verbose=-1)

lgbm = lgbm.fit(x_train, y_train)

train_test(rid, x_train, y_train, x_test, y_test)
xgbREG = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:squarederror', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)

xgbREG = xgbREG.fit(x_train, y_train)

train_test(xgbREG, x_train, y_train, x_test, y_test)
stack = StackingCVRegressor(regressors=(GB, BRR, lasso, elnet, rid, lgbm, xgbREG),

                            meta_regressor=xgbREG, random_state=10, use_features_in_secondary=True)

stack = stack.fit(x_train, y_train)

train_test(stack, x_train, y_train, x_test, y_test)
#coefficients can be improved, not using the stacked and lasso

def blended_predict(X):

    return ((2 * GB.predict(X)) +

            (2 * BRR.predict(X)) +

            #(1 * lasso.predict(X)) +

            (1 * elnet.predict(X)) +

            (1 * rid.predict(X)) +

            (1 * lgbm.predict(X)) +

            (3 * xgbREG.predict(X))) / 10
print("train")

get_score(blended_predict(x_train), y_train)    

print("test")

get_score(blended_predict(x_test), y_test)    
#training final models used in the blending on all the training data

GB = GB.fit(X_train_all, y_train_all)

BRR = BRR.fit(X_train_all, y_train_all)

elnet = elnet.fit(X_train_all, y_train_all)

rid = rid.fit(X_train_all, y_train_all)

lgbm = lgbm.fit(X_train_all, y_train_all)

xgbREG = xgbREG.fit(X_train_all, y_train_all)
get_score(blended_predict(X_train_all), y_train_all)
sub_labels = np.expm1(blended_predict(X_sub))
pd.DataFrame({'Id': sub_id, 'SalePrice': sub_labels}).to_csv('2019-12-16.csv', index=False)