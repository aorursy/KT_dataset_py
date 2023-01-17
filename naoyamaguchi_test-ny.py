import numpy as np

import pandas as pd

import datetime

import random

import os

from pandas import DataFrame, Series

from sklearn.preprocessing import StandardScaler

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold, KFold



from sklearn.svm import SVC

from sklearn import tree

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import log_loss

from tensorflow.keras.models import model_from_json

from rgf.sklearn import FastRGFClassifier

import catboost as cb

import xgboost as xgb



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from tqdm import tqdm_notebook as tqdm

import datetime as dt



import optuna.integration.lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor



from sklearn.model_selection import RandomizedSearchCV

from scipy import stats

from statistics import mean, median, variance, stdev



from sklearn.preprocessing import quantile_transform

import category_encoders as ce

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from xgboost import XGBClassifier, XGBRegressor

epsilon = 1e-7



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



# 乱数シード固定

seed_everything(71)

input_path='/kaggle/input/exam-for-students20200527'
df_train = pd.read_csv(input_path+'/train.csv')

df_test = pd.read_csv(input_path+'/test.csv')

df_city = pd.read_csv(input_path+'/city_info.csv')

df_station = pd.read_csv(input_path+'/station_info.csv')
print('df_train len:{}'.format(len(df_train)))

print('df_test len:{}'.format(len(df_test)))

len_train = len(df_train)

print(len_train)
#ターゲット処理

y_train = df_train['TradePrice']

df_train.drop('TradePrice',axis=1,inplace=True)

print(len(y_train))
#訓練、テストデータ結合

df = pd.concat([df_train,df_test],axis=0)

print(len(df))
#データ結合

df = pd.merge(df,df_station,left_on='NearestStation',right_on='Station',how='left')

df.rename(columns={'Latitude':'st_Latitude','Longitude':'st_Longitude'},inplace=True)



df = pd.merge(df,df_city,on=['Prefecture','Municipality'],how= 'left')

df.rename(columns={'Latitude':'ct_Latitude','Longitude':'ct_Longitude'},inplace=True)



df.drop('Station',inplace=True,axis=1)

df.info()
#オブジェクト型取得

obj_col_name = []

for col in df_train.columns:

    if df[col].dtype == 'object':

        obj_col_name.append(col)

        

        print(col, df[col].nunique())



#数値型列名取得

num_col_name = []

for col in df.columns:

    if df[col].dtype in ['float64','int64']:

        num_col_name.append(col)

        

        print(col, df[col].nunique())
#ラベルエンコディング



oe = OrdinalEncoder(cols=obj_col_name, return_df=False)

df[obj_col_name] = oe.fit_transform(df[obj_col_name])



#df_train = df_all.iloc[:df_train.shape[0],:].reset_index(drop=True)

#df_test = df_all.iloc[df_test.shape[0]:,:].reset_index(drop=True)
#カウントエンコーディング



for col in obj_col_name:

    df_cnt = df[col].value_counts()

    

    df[col+'_cnt'] = df[col].map(df_cnt)
df.fillna(-99999,inplace=True)

#add later

df.drop(['id','TimeToNearestStation','MaxTimeToNearestStation'],inplace=True,axis=1)

df['h_age'] = 2020 - df['BuildingYear']
tokyo_latitude = df_station[df_station['Station']=='Tokyo']['Latitude'].values

tokyo_longitude = df_station[df_station['Station']=='Tokyo']['Longitude'].values



df['st_tokyo_ds'] = np.sqrt((df['st_Latitude'] - tokyo_latitude)**2 + (df['st_Longitude'] - tokyo_longitude)**2)
df.head()
#データ分割

df_train = df[:len_train]

df_test = df[len_train:]

print('df_train len:{}'.format(len(df_train)))

print('df_test len:{}'.format(len(df_test)))

print('y_train len:{}'.format(len(y_train)))

X_train = df_train

X_test = df_test
X_train.reset_index(drop=True,inplace=True)

y_train.reset_index(drop=True,inplace=True)

X_test.reset_index(drop=True,inplace=True)

print('X_train len:{}'.format(len(X_train)))

print('X_test len:{}'.format(len(X_test)))

print('y_train len:{}'.format(len(y_train)))
X_train.head()
#モデル作成

'''

lgb_params = {'boosting_type': 'gbdt','objective': 'regression',

              'metric': 'rmse','learning_rate':0.05,

              'min_child_samples': 20,

              'min_child_weight' : 0.001,

              'min_split_gain' : 0.0,

              'max_depth' : -1,

              'reg_alpha' : 1.0,

              'reg_lambda' : 1.0,

             'num_leaves' : 50,

             'num_iterations' : 100,

             'colsample_bytree' : 0.9,

             'num_parallel_tree' :1,

             'subsample_for_bin': 200000,

              'n_estimators' : 100

             }

'''



lgb_params = {'boosting_type': 'gbdt','objective': 'regression',

              'metric': 'rmse','learning_rate':'0.05',

              'min_child_samples':'20',

              'max_depth' : -1,

             'num_leaves' : 50,

             'num_iterations' : 100,

             'colsample_bytree' : 0.8,

             'num_parallel_tree' :1}



#lgbm = LGBMRegressor(**lgb_params)
execution_KF = True



if execution_KF == True:

    

    skf = KFold(n_splits=5, shuffle=True, random_state=71)

    for i,(idx_train, idx_valid) in enumerate((skf.split(X_train))):

        X_train_, X_valid = X_train.iloc[idx_train],X_train.iloc[idx_valid]

        y_train_, y_valid = y_train.iloc[idx_train],y_train.iloc[idx_valid]

    

        lgb_train = lgb.Dataset(X_train_, np.log1p(y_train_))

        lgb_valid = lgb.Dataset(X_valid, np.log1p(y_valid))

    

        lgb_model = lgb.train(lgb_params,

                        lgb_train,

                        #num_boost_round=10000,

                        valid_sets=(lgb_train,lgb_valid),

                        verbose_eval=0)



    #    lgb_model = lgbm.fit(X_train_, np.log1p(y_train_),

    #                        early_stopping_rounds = 20,

    #                        eval_metric = 'RMSE',

    #                        eval_set = [(X_valid, np.log1p(y_valid))])



if execution_KF == False:

    lgb_train = lgb.Dataset(X_train, np.log1p(y_train))

#    lgb_valid = lgb.Dataset(X_valid, np.log1p(y_valid))

    

    lgb_model = lgb.train(lgb_params,

                        lgb_train,

                        num_boost_round=10000)
lgb_pred = np.expm1(lgb_model.predict(X_test))
# GradientBoosting model

gb_model = GradientBoostingRegressor() 

gb_model.fit(X_train, np.log1p(y_train))

gb_pred = np.expm1(gb_model.predict(X_test))
# XGBoost model



xgb_model = xgb.XGBRegressor()

xgb_model.fit(X_train, np.log1p(y_train))

xgb_pred = np.expm1(xgb_model.predict(X_test))
#pred_test = (lgb_pred + gb_pred + xgb_pred) / 3

#pred_test = gb_pred

pred_test = lgb_pred
#perm = PermutationImportance(xgb_model, random_state = 1).fit(X_train, np.log1p(y_train))

#eli5.show_weights(perm, feature_names = X_train.columns.tolist())
submission = pd.read_csv(input_path+'/sample_submission.csv', index_col=0)



submission.TradePrice = np.round(pred_test,-5)

submission.to_csv('submission.csv')