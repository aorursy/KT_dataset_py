import numpy as np

import pandas as pd

import datetime



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import xgboost as xgb



import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_columns', 200)
df_hist_trans = pd.read_csv('../input/dsa-comp4/transacoes_historicas.csv'

                            ,parse_dates=['purchase_date']

                            ,dtype = {

                                'city_id': np.int16

                                ,'installments': np.int16

                                ,'merchant_category_id': np.int16

                                ,'month_lag': np.int8

                                ,'purchase_amount': np.float32

                                ,'state_id': np.int8

                                ,'subsector_id': np.int8

                            }) 



df_new_merchant_trans = pd.read_csv('../input/dsa-comp4/novas_transacoes_comerciantes.csv'

                            ,parse_dates=['purchase_date']

                            ,dtype = {

                                'city_id': np.int16

                                ,'installments': np.int16

                                ,'merchant_category_id': np.int16

                                ,'month_lag': np.int8

                                ,'purchase_amount': np.float32

                                ,'state_id': np.int8

                                ,'subsector_id': np.int8

                            })   



df_train = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_treino.csv'

                       ,parse_dates=['first_active_month']

                       ,dtype = {

                                'feature_1': np.int8

                                ,'feature_2': np.int8

                                ,'feature_3': np.int8

                            })



df_test = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_teste.csv'

                        ,parse_dates=['first_active_month']

                        ,dtype = {

                                'feature_1': np.int8

                                ,'feature_2': np.int8

                                ,'feature_3': np.int8

                            })



df_comerciantes = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/comerciantes.csv')
df_hist_trans.head()
df_new_merchant_trans.head()
df_train.head()
df_test.head()
df_comerciantes.head()
pd.DataFrame(df_hist_trans.isnull().sum().sort_values(ascending=False)).head(4)
pd.DataFrame(df_new_merchant_trans.isnull().sum().sort_values(ascending=False)).head(4)
pd.DataFrame(df_train.isnull().sum().sort_values(ascending=False)).head(4)
pd.DataFrame(df_test.isnull().sum().sort_values(ascending=False)).head(4)
pd.DataFrame(df_comerciantes.isnull().sum().sort_values(ascending=False)).head(4)
df_hist_trans['category_2'].value_counts()
df_new_merchant_trans['category_2'].value_counts()
df_hist_trans['category_3'].value_counts()
df_new_merchant_trans['category_3'].value_counts()
df_hist_trans['merchant_id'].value_counts().head(5)
df_new_merchant_trans['merchant_id'].value_counts().head(5)
for df in [df_train, df_test]:

    df['year'] = df['first_active_month'].dt.year

    df['month'] = df['first_active_month'].dt.month

    df['dayofweek'] = df['first_active_month'].dt.dayofweek
df_train.head()
features = ['feature_1', 'feature_2', 'feature_3', 'year', 'month', 'dayofweek']



param = {

    'objective': 'reg:linear'

    ,'booster' : "gbtree"

    ,'eta': 0.01

    ,'max_depth':10

    ,'subsample':0.9

    ,'colsample_bytree':0.7

    ,'silent' : 1

    ,'eval_metric': 'rmse'

    #,'tree_method':'gpu_hist'

    #,'predictor':'gpu_predictor'

}



X_train, X_test, y_train, y_test = train_test_split(df_train[features], df_train['target'], test_size = 0.3, random_state = 42)



dtrain = xgb.DMatrix(X_train, y_train)

dvalid = xgb.DMatrix(X_test, y_test)



watchlist = [(dtrain, 'train'), (dvalid, 'eval')]





gbm = xgb.train(

            param, 

            dtrain, 

            7000,

            evals=watchlist,

            early_stopping_rounds=100, 

            verbose_eval=100

)
predictions = gbm.predict(xgb.DMatrix(df_test[features]))

predictions
sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions

sub_df.to_csv("../submission.csv", index=False)