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
        'eta': 0.1, # Step size shrinkage used in update to prevents overfitting
        'max_depth': 10,
        'subsample': 0.6, # sample of rows
        'colsample_bytree': 0.6, # sample of features
#         'alpha':0.001, 
        'lambda':1, # l2 regu
        'random_state': 42,
        'silent':True
        
    }
    xgb_train_data = xgb.DMatrix(X_train, y_train)
    xgb_val_data = xgb.DMatrix(X_valid, y_valid)
    
    model = xgb.train(params, xgb_train_data,
          num_boost_round=1000,
          evals= [(xgb_train_data, 'train'), (xgb_val_data, 'valid')],
          early_stopping_rounds=10, 
          verbose_eval=20
         )
    return model

def submit():
    test_matrix = xgb.DMatrix(X_test)
    y_pred = clf.predict(test_matrix,ntree_limit=clf.best_ntree_limit)
    df_test['PredictedLogRevenue'] = y_pred
    submit = df_test[['PredictedLogRevenue','fullVisitorId']].groupby('fullVisitorId').PredictedLogRevenue.sum().reset_index()
    submit.to_csv('submit.csv',index=False)
    
    test(y_pred)

    
def test(predict):
    y_test = df_test['totals_transactionRevenue']
    print(rmse(y_test,predict))

def prepare_data(df_train,df_test,
                 del_col=['fullVisitorId','visitId','visitStartTime','date']):
    df_train = df_train.sort_values('date').copy()
    
    df_train = df_train.drop(del_col,axis=1).copy()
    df_test = df_test.drop(del_col,axis=1).copy()
    
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
    
    
encode_data(verbose=True)
prepare_data(df_train_encoded,df_test_encoded,del_col=['fullVisitorId','visitId',
            'visitStartTime','date','totals_totalTransactionRevenue','totals_transactions'])

clf = run_xgb()
# try to find a good validation set
# Why our score so different with the leader board?
# check with the target in dataset first
xgb.plot_importance(clf,importance_type='gain')
plt.title('Gain Feature important')
xgb.plot_importance(clf,importance_type='cover')
plt.title('Cover Feature important')
xgb.plot_importance(clf,importance_type='weight')
plt.title('Weight Feature important')
submit()





