# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from os.path import join as pjoin

data_root = '../input/make-data-ready'
print(os.listdir(data_root))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor as RFF
import xgboost as xgb

from pprint import pprint
import math

from scipy.stats import kurtosis, skew


import seaborn as sns
import matplotlib.pyplot as plt
import shap
plt.rcParams['figure.figsize'] = (12,6)
def load_data(data='train',n=2):
    df = pd.DataFrame()
    for i in range(n) :
        if data=='train':
            if i > 8 :
                break
            dfpart = pd.read_pickle(pjoin(data_root,f'train_{i}.pkl'))
        elif data=='test':
            if i > 2 :
                break
            dfpart = pd.read_pickle(pjoin(data_root,f'test_{i}.pkl'))
        df = pd.concat([df,dfpart])
        del dfpart
    return df
        
df_train = load_data(n=9)
df_test = load_data('test',n=4)
print(f'# of columns has na value: {(df_test.isnull().sum().sort_values(ascending=False) > 0).sum()}')
def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)

def split_data(df=df_train,rate=.8):
    # sort the date first
    df = df.sort_values('date').copy()
    
    df.drop(['fullVisitorId','visitId','visitStartTime'],axis=1,inplace=True)
    df['Revenue'] = np.log1p(df['Revenue'])
    
    global X_train,X_valid,y_train,y_valid
    
    n_train = int(len(df)*rate)
    X_train = df.drop(['Revenue','date'],axis=1).iloc[:n_train]
    X_valid = df.drop(['Revenue','date'],axis=1).iloc[n_train:]
    
    y_train = df['Revenue'].iloc[:n_train]
    y_valid = df['Revenue'].iloc[n_train:]
    
    print(X_train.shape,X_valid.shape)
    
    

def encode_data(verbose=False):
    global df_train_encoded,df_test_encoded
    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()
    for col in df_train.columns:
        if df_train_encoded[col].dtype == 'object' and col not in ['fullVisitorId','visitId','visitStartTime','date']:
            if verbose:
                print(col)
            lb = LabelEncoder()
            lb.fit( list(df_train_encoded[col].unique()) + list(df_test_encoded[col].unique()))
            df_train_encoded[col] = lb.transform(df_train_encoded[col])
            df_test_encoded[col] = lb.transform(df_test_encoded[col])
        
def run_xgb():
   
    params = {
        'objective':'reg:linear',
        'eval_metric':'rmse',
        'learning_rate':.01,
        'eta': 0.15, # Step size shrinkage used in update to prevents overfitting
#         'max_depth': 10, # V3 : 1.0471 on LB
#         'max_depth':5, # V5 : 0.9331 on LB
        'subsample': 0.6, # sample of rows
        'colsample_bytree': 0.6, # sample of features
#         'alpha':0.001, 
        'lambda':1, # l2 regu
        'random_state': 42,
        'silent':True
        
    }
    
    
    # got params from https://www.kaggle.com/kailex/group-xgb-for-gstore-v2
    params['n_thread'] = -1
    params['max_depth'] = 8
    params['min_child_weight'] = 100
    params['gamma'] = 5
    params['subsample'] = 1
    params['colsample_bytree'] = .95
    params['colsample_bylevel'] = 0.35
    params['alpha'] = 25
    params['lambda'] = 25
    
    xgb_train_data = xgb.DMatrix(X_train, y_train)
    xgb_val_data = xgb.DMatrix(X_valid, y_valid)
    
    model = xgb.train(params, xgb_train_data,
#           num_boost_round=1000, # V3 : 1.0471 on LB
#           num_boost_round=200, # 1.0471 on LB
          num_boost_round = 200,
          evals= [(xgb_train_data, 'train'), (xgb_val_data, 'valid')],
#           early_stopping_rounds=10, # V11 0.9301 on LB
          early_stopping_rounds=50, 
          verbose_eval=20
         )
    return model

def submit():
    test_matrix = xgb.DMatrix(X_test)
    y_pred = clf.predict(test_matrix,ntree_limit=clf.best_ntree_limit)
    df_test['PredictedLogRevenue'] = y_pred
    engineer_prediction
    print('rmse after engineer prediction')
    print(rmse(y_pred,df_test['PredictedLogRevenue']))
    submit = df_test[['PredictedLogRevenue','fullVisitorId']].groupby('fullVisitorId').PredictedLogRevenue.sum().reset_index()
    submit.to_csv('submit.csv',index=False)
    
    test(y_pred)
    
    
def engineer_prediction(df_test):
    df_test[df_test['totals_hits'] == 1].PredictedLogRevenue = 0
    df_test[df_test['totals_timeOnSite'] == 0].PredictedLogRevenue = 0
    df_test[df_test['totals_bouces'] == 1].PredictedLogRevenue = 0
    return dftest
    
    
def test(predict):
    y_test = np.log1p(df_test['totals_transactionRevenue'])
    print(rmse(y_test,predict))

def prepare_data(df_train,df_test,
                 del_col=['fullVisitorId','visitId','visitStartTime','date'],to_log=None):
    df_train = df_train.sort_values('date').copy()
    
    df_train = df_train.drop(del_col,axis=1).copy()
    df_test = df_test.drop(del_col,axis=1).copy()
    
    # Log some column
    if to_log is not None:
        df_train[to_log] = np.log1p(df_train[to_log])
        df_test[to_log] = np.log1p(df_test[to_log])
    
    # totals_transactionRevenue
    df_train['totals_transactionRevenue'] = np.log1p(df_train['totals_transactionRevenue'])
    df_test['totals_transactionRevenue'] = np.log1p(df_test['totals_transactionRevenue'])
    
    global X_train,X_valid,y_train,y_valid,X_test,y_test
    # 80/20 : train/valid
    n_train = int(len(df_train)*.8)
    
    # split
    X_train = df_train.drop(['totals_transactionRevenue'],axis=1).iloc[:n_train]
    X_valid = df_train.drop(['totals_transactionRevenue'],axis=1).iloc[n_train:]
    
    y_train = df_train['totals_transactionRevenue'].iloc[:n_train]
    y_valid = df_train['totals_transactionRevenue'].iloc[n_train:]
    
    X_test = df_test.drop(['totals_transactionRevenue'],axis=1)
    y_test = df_test['totals_transactionRevenue']
    
    
def feature_engineering(df):
    df = df.copy()
    # Copy from : https://www.kaggle.com/qnkhuat/base-model-v2-with-with-full-features/edit
    
    # time based
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['weekofyear'] = df['date'].dt.weekofyear
    
    df['browser_category'] = df['device_browser'] + '_' + df['device_deviceCategory']
    df['browser_operatingSystem'] = df['device_browser'] + '_' + df['device_operatingSystem']

    df['month_unique_user_count'] = df.groupby('month')['fullVisitorId'].transform('nunique')
    df['day_unique_user_count'] = df.groupby('day')['fullVisitorId'].transform('nunique')
    df['weekday_unique_user_count'] = df.groupby('weekday')['fullVisitorId'].transform('nunique')
    df['weekofyear_unique_user_count'] = df.groupby('weekofyear')['fullVisitorId'].transform('nunique')
    
    
    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('mean')
    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('mean')
    
    df['mean_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('mean')
    df['sum_hits_per_day'] = df.groupby(['day'])['totals_hits'].transform('sum')
    
    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('mean')

    df['sum_pageviews_per_region'] = df.groupby('geoNetwork_region')['totals_pageviews'].transform('sum')
    df['count_pageviews_per_region'] = df.groupby('geoNetwork_region')['totals_pageviews'].transform('count')
    df['mean_pageviews_per_region'] = df.groupby('geoNetwork_region')['totals_pageviews'].transform('mean')
    
    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('mean')

    df['sum_hits_per_region'] = df.groupby('geoNetwork_region')['totals_hits'].transform('sum')
    df['count_hits_per_region'] = df.groupby('geoNetwork_region')['totals_hits'].transform('count')
    df['mean_hits_per_region'] = df.groupby('geoNetwork_region')['totals_hits'].transform('mean')

    df['sum_hits_per_country'] = df.groupby('geoNetwork_country')['totals_hits'].transform('sum')
    df['count_hits_per_country'] = df.groupby('geoNetwork_country')['totals_hits'].transform('count')
    df['mean_hits_per_country'] = df.groupby('geoNetwork_country')['totals_hits'].transform('mean')
    
    df['user_pageviews_sum'] = df.groupby('fullVisitorId')['totals_pageviews'].transform('sum')
    df['user_hits_sum'] = df.groupby('fullVisitorId')['totals_hits'].transform('sum')
    
    df['user_pageviews_count'] = df.groupby('fullVisitorId')['totals_pageviews'].transform('count')
    df['user_hits_count'] = df.groupby('fullVisitorId')['totals_hits'].transform('count')

    
    df['user_pageviews_sum_to_mean'] = df['user_pageviews_sum'] / df['user_pageviews_sum'].mean()
    df['user_hits_sum_to_mean'] = df['user_hits_sum'] / df['user_hits_sum'].mean()

    df['user_pageviews_to_region'] = df['user_pageviews_sum'] / df['mean_pageviews_per_region']
    df['user_hits_to_region'] = df['user_hits_sum'] / df['mean_hits_per_region']
    
    return df
df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)
encode_data(verbose=True)
prepare_data(df_train_encoded,df_test_encoded,del_col=['fullVisitorId','visitId',
            'visitStartTime','date'],to_log=['visitNumber','totals_hits','totals_pageviews',
            'totals_sessionQualityDim','totals_timeOnSite','trafficSource_keyword'])
# prepare_data(df_train_encoded,df_test_encoded,del_col=['fullVisitorId','visitId',
#             'visitStartTime','date'])


clf = run_xgb()
# try to find a good validation set
# Why our score so different with the leader board?
# check with the target in dataset first
xgb.plot_importance(clf,importance_type='gain',max_num_features=20)
plt.title('Gain Feature important')
xgb.plot_importance(clf,importance_type='cover',max_num_features=20)
plt.title('Cover Feature important')
xgb.plot_importance(clf,importance_type='weight',max_num_features=20)
plt.title('Weight Feature important')
submit()





