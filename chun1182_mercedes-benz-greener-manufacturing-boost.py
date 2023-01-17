!pip install category_encoders
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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df_train = pd.read_csv("../input/train.csv.zip")

df_test = pd.read_csv("../input/test.csv.zip")

print("Train shape : ", df_train.shape)

print("Test shape : ", df_test.shape)



df_train.head()
df_train.describe(include='all')
df_train['y'].hist(bins=20)
df_test.head()
df_test.describe(include='all')
df_train.columns
'''

import category_encoders as ce

list_cols = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

target_columns = df_train.columns[2:]

# OneHotEncodeしたい列を指定。Nullや不明の場合の補完方法も指定。

ce_ohe = ce.OneHotEncoder(cols=list_cols)

train_onehot = ce_ohe.fit_transform(df_train[target_columns])

test_onehot = ce_ohe.transform(df_test[target_columns])

test_onehot.describe()

'''
from sklearn import preprocessing

for column in ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']:

    le = preprocessing.LabelEncoder()

    le.fit(list(df_train[column]) + list(df_test[column]))

    df_train[column] = le.transform(df_train[column])

    df_test[column] = le.transform(df_test[column])



target_columns = df_train.columns[2:]

train_onehot = df_train[target_columns]

test_onehot = df_test[target_columns]
from sklearn.decomposition import PCA

pca2 = PCA(n_components=5)

pca2_results = pca2.fit_transform(df_train.drop(["y"], axis=1))

df_train['pca0']=pca2_results[:,0]

df_train['pca1']=pca2_results[:,1]

df_train['pca2']=pca2_results[:,2]

df_train['pca3']=pca2_results[:,3]

df_train['pca4']=pca2_results[:,4]

pca2_results = pca2.transform(df_test)

df_test['pca0']=pca2_results[:,0]

df_test['pca1']=pca2_results[:,1]

df_test['pca2']=pca2_results[:,2]

df_test['pca3']=pca2_results[:,3]

df_test['pca4']=pca2_results[:,4]
from sklearn.model_selection import train_test_split



y_train = df_train["y"]

y_mean = np.mean(y_train)

df_train.drop('y', axis=1, inplace=True)
'''

import xgboost as xgb

xgb_params = {

    'eta': 0.02,

    'max_depth': 4,

    'subsample': 0.95,

    # 'colsample_bytree': 0.95,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean,

    'silent': 1

}



dtrain = xgb.DMatrix(df_train, y_train)

dtest = xgb.DMatrix(df_test)

'''
'''

cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

   verbose_eval=True, show_stdv=False)



num_boost_rounds = len(cv_result)

print(num_boost_rounds)

# num_boost_round = 489

num_boost_round = 20

'''
'''

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

'''
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score



gbk = GradientBoostingRegressor()

gbk.fit(df_train, y_train)

#y_pred = gbk.predict(x_val)



#r2 = r2_score(y_val, y_pred)

#print(r2)
'''

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

# データセットを生成する

lgb_train = lgb.Dataset(x_train, y_train)

lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)



# LightGBM のハイパーパラメータ

lgbm_params = {'objective': 'regression','metric': 'rmse',}

# 上記のパラメータでモデルを学習する

model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)



# テストデータを予測する

y_pred = model.predict(x_val, num_iteration=model.best_iteration)



# RMSE を計算する

mse = mean_squared_error(y_val, y_pred)

rmse = np.sqrt(mse)



r2 = r2_score(y_val, y_pred)

print(r2)

'''
preds = gbk.predict(df_test)
submmision = pd.read_csv("../input/sample_submission.csv.zip")

submmision["y"] = preds

submmision.to_csv("benz_test.csv", index=False)