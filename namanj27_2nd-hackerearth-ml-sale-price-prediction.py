# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/hackerearth-ml-slashing-prices-for-biggest-sale/Train.csv')

test_df = pd.read_csv('../input/hackerearth-ml-slashing-prices-for-biggest-sale/Test.csv')

sample = pd.read_csv('../input/hackerearth-ml-slashing-prices-for-biggest-sale/sample_submission.csv')
train_df.head()
from scipy.stats import norm

sns.distplot(train_df["Low_Cap_Price"],fit=norm)

mu,sigma= norm.fit(train_df['Low_Cap_Price'])

print("mu {}, sigma {}".format(mu,sigma))
# Removing skewness

########## REMOVING SKEWEENESS ###########

train_df['Low_Cap_Price']=np.log1p(train_df['Low_Cap_Price'])

sns.distplot(train_df['Low_Cap_Price'],fit=norm)

mu,sigma= norm.fit(train_df['Low_Cap_Price'])

print("mu {}, sigma {}".format(mu,sigma))
test_df['Low_Cap_Price'] = -1
df = pd.merge(train_df, test_df, how='outer')

df.head()
df['Date'] = pd.to_datetime(df['Date'])
# df['month'] = df['Date'].dt.month

df['year'] = df['Date'].dt.year

# df['dates'] = df['Date'].dt.day
df['State_of_Country'].corr(df['Low_Cap_Price'])
numeric_col = ['Demand','High_Cap_Price']

skew=df[numeric_col].skew()

skew
from scipy.special import boxcox1p

lam=0.15

for i in skew.index:

    df[i]=np.log1p(df[i])
# One Hot

df = pd.get_dummies(df, columns=['State_of_Country','Market_Category','Product_Category','Grade','year'])

# test_df = pd.get_dummies(test_df, columns=['State_of_Country','Market_Category','Product_Category','Grade'])




#Normalize Demand and High_Cap_Price

df['Demand'] = (df['Demand'] - np.mean(df['Demand'])) / (np.max(df['Demand']) - np.min(df['Demand']))

# test_df['Demand'] = (test_df['Demand'] - np.mean(test_df['Demand'])) / (np.max(test_df['Demand']) - np.min(test_df['Demand']))



df['High_Cap_Price'] = (df['High_Cap_Price'] - np.mean(df['High_Cap_Price'])) / (np.max(df['High_Cap_Price']) - np.min(df['High_Cap_Price']))

# test_df['High_Cap_Price'] = (test_df['High_Cap_Price'] - np.mean(test_df['High_Cap_Price'])) / (np.max(test_df['High_Cap_Price']) - np.min(test_df['High_Cap_Price']))
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import ElasticNet,BayesianRidge

from sklearn.preprocessing import RobustScaler
features = [c for c in df.columns if c not in ['Item_Id','Date','Low_Cap_Price']]

features
train_df = df[df['Low_Cap_Price']!=-1]

test_df = df[df['Low_Cap_Price']==-1]
print(len(train_df.columns))

print(len(test_df.columns))
X = train_df[features]

y = train_df['Low_Cap_Price']



x_test = test_df[features]
xgb = XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1, 

                    importance_type='gain', interaction_constraints=None,

                     min_child_weight=1, missing=None, monotone_constraints=None,

                     n_estimators=700, n_jobs=-1, nthread=-1, num_parallel_tree=1

                    )

xgb.fit(X, y)

pred_1 = xgb.predict(x_test)

pred_1[pred_1 < 0] = 1
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.03, n_estimators=2200,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



model_lgb.fit(X,y)

pred_2 = model_lgb.predict(x_test)

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5,)

GBoost.fit(X, y)

pred_4 = GBoost.predict(x_test)

#XGBOOST

model=XGBRegressor(n_estimators=2200,learning_rate=0.05)

model.fit(X,y)



pred_6 = model.predict(x_test)

# pred_1 = 0.7*(pred_1 + pred_2)/2.0 + 0.3*(pred_5 + pred_4)/2.0
pred_1 = np.expm1(pred_1)

pred_2 = np.expm1(pred_2)

# pred_3 = np.expm1(pred_3)

pred_4 = np.expm1(pred_4)

# pred_5 = np.expm1(pred_5)

pred_6 = np.expm1(pred_6)

# pred_7 = np.expm1(pred_7)
pred_1 = (pred_1 + pred_6 + pred_2 + pred_4)/4.0
sub = pd.DataFrame({'Item_Id':test_df.Item_Id.values,

                   'Low_Cap_Price':pred_1})

sub.to_csv('submission_ensemble.csv', index=False)