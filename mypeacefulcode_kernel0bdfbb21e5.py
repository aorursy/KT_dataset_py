# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import math

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy import stats

from sklearn import preprocessing



df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
for df in [df_train, df_test]:

  df['date'] = df['date'].apply(lambda x:x[:6]).astype(int)

  df['total_rooms'] = df['bedrooms'] + df['bathrooms']

  df['sqft_total'] = df['sqft_living'] + df['sqft_lot']

  df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

  df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15']

  df['sqft_grade'] = df['sqft_living'] * df['grade']

  df['renovated_yn'] = df['yr_renovated'].apply(lambda x: 0 if x == 0 else 1)

  df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

  df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])

  df['has_basement'] = df['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)
df_train['price_per_dist'] = 0

for index, row in df_train.iterrows():

  l1 = [row['lat'], row['long']]

  dist_index = df_train[['lat', 'long']].sub(np.array(l1)).pow(2).sum(1).pow(0.5).sort_values(ascending=True).index[:20]

  df_train.iloc[index, df_train.columns.get_loc('price_per_dist')] = df_train.loc[dist_index]['price'].sum()/df_train.loc[dist_index]['sqft_living'].sum()
df_test['price_per_dist'] = 0

for index, row in df_test.iterrows():

  dist_index = df_train[['lat', 'long']].sub(np.array(l1)).pow(2).sum(1).pow(0.5).sort_values(ascending=True)[:1].index

  df_test.iloc[index, df_test.columns.get_loc('price_per_dist')] = df_train.loc[dist_index]['price_per_dist'].tolist().pop()
df_train = df_train.drop(df_train[df_train['id'].isin([8912])].index)

df_train['price'] = np.log1p(df_train['price'])
skew_index = ['sqft_living','sqft_lot','sqft_above','sqft_basement','sqft_living15','sqft_lot15', \

              'sqft_total','sqft_ratio', 'sqft_grade', 'price_per_dist']

for df in [df_train, df_test]:

  for i in skew_index:

    df[i] = np.log1p(df[i])
from datetime import datetime



from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, Ridge, Lasso, ElasticNet

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error, mean_absolute_error



from mlxtend.regressor import StackingCVRegressor

from sklearn.model_selection import GridSearchCV



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
seed = 2019

np.random.seed(seed)
# rmsle

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))





# build our model scoring function

def cv_rmse(model, X, y):

    rmse = np.sqrt(-cross_val_score(model, X, y,

                                    scoring="neg_mean_squared_error",

                                    cv=kfolds))

    return (rmse)
kfolds = KFold(n_splits=5, shuffle=True, random_state=seed)



X_train = df_train.drop(['id','price'], axis=1)

y_train = df_train['price']

X_test = df_test.drop(['id'], axis=1)

test_ids = df_test['id']
ridge =Ridge(alpha=0.7, random_state=seed)



lasso = Lasso(alpha=0.01, random_state=seed)

                                        

svr = SVR(C=60, epsilon=0.2, gamma=0.0001)



lgbm = LGBMRegressor(objective='regression', num_leaves=80,

                     learning_rate=0.01, n_estimators=1000,

                     max_bin=100, max_depth=7)





xgb = XGBRegressor(learning_rate=0.025, n_estimators=1000,

                   max_depth=6, subsample=0.7, base_score=np.mean(y_train),

                   objective='reg:linear') 



rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=10, min_samples_split=10, random_state=seed)



stack = StackingCVRegressor(

    regressors=(xgb, rf, ridge, lasso, lgbm), cv=5,

    meta_regressor=lgbm, use_features_in_secondary=True

)
print('START Fit')

print(datetime.now(), 'StackingCVRegressor')

model = stack.fit(np.array(X_train), np.array(y_train))
def blend_models_predict(X):

    return model.predict(np.array(X))
print('RMSLE score on train data:')

print(rmsle(y_train, blend_models_predict(X_train)))
print('Predict submission', datetime.now(),)

print(os.listdir("./"))

preds = np.floor(np.expm1(blend_models_predict(X_test)))

submission = pd.DataFrame({'id': test_ids, 'price': preds})

submission.to_csv('./submission.csv', index=False)