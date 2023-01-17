import pandas as pd

pd.set_option('display.max_columns', 500)



from pandas import Series

import numpy as np

import scipy as sp

import sklearn

import copy



import random as rnd

import time



import warnings

warnings.filterwarnings('ignore')

 

import xgboost as xgb

from xgboost import XGBRegressor

import lightgbm as lgb



from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix

import missingno as msno



#Configure Visualization Defaults

%matplotlib inline

mpl.style.use('seaborn')
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
## 년월만 추출

df_train['YYYYMM'] = df_train.date.str[:6]

df_test['YYYYMM'] = df_test.date.str[:6]
## 년월은 숫자로 변환

df_train['YYYYMM'] = df_train['YYYYMM'].astype('int')

df_test['YYYYMM'] = df_test['YYYYMM'].astype('int')
## zipcode별 평당 평균땅값

df_zipsum = df_train[['zipcode','sqft_lot']].groupby('zipcode', as_index = False).sum()

df_zipcount = df_train[['zipcode','price']].groupby('zipcode', as_index=False).sum()

df_zip_avg = pd.merge(df_zipsum, df_zipcount, on='zipcode')

df_zip_avg['zip_avg_price'] = df_zip_avg['price'] / df_zip_avg['sqft_lot'] 



df_train = pd.merge(df_train, df_zip_avg[['zipcode','zip_avg_price']], on ='zipcode', how='left')

df_test =  pd.merge(df_test,  df_zip_avg[['zipcode','zip_avg_price']], on ='zipcode', how='left')
# zipcode별 평당 평균땅값 * 부지

df_train['zipvalue'] =  df_train['zip_avg_price']*df_train['sqft_lot']

df_test['zipvalue'] =  df_test['zip_avg_price']*df_test['sqft_lot']
## zipcode별 평균 집값

df_zip_mean = df_train[['zipcode','price']].groupby('zipcode', as_index=False).mean()

df_zip_mean.rename(columns={'price':'zip_avg_house_price'}, inplace=True)

df_train = pd.merge(df_train, df_zip_mean[['zipcode','zip_avg_house_price']], on ='zipcode', how='left')

df_test = pd.merge(df_test, df_zip_mean[['zipcode','zip_avg_house_price']], on ='zipcode', how='left')
#지상지하 합면적

df_train['totallot'] = df_train['sqft_above'] + df_train['sqft_basement']

df_test['totallot'] = df_test['sqft_above'] + df_test['sqft_basement']
!pip install uszipcode
# city from zipcode

from uszipcode import SearchEngine

search = SearchEngine(simple_zipcode=True)



states =[]

for i in df_train['zipcode'] :

    states.append(search.by_zipcode(i).values()[3])

df_train['states'] = states



states =[]

for i in df_test['zipcode'] :

    states.append(search.by_zipcode(i).values()[3])

df_test['states'] = states
# states별 평균집값

df_states_mean = df_train[['states','price']].groupby('states', as_index=False).mean()

df_states_mean.rename(columns={'price':'states_avg_house_price'}, inplace=True)

df_train = pd.merge(df_train, df_states_mean[['states','states_avg_house_price']], on ='states', how='left')

df_test = pd.merge(df_test, df_states_mean[['states','states_avg_house_price']], on ='states', how='left')

## 타겟레이블 log변환

df_train['price'] = np.log1p(df_train['price'])
## city 정보 one-hot encoding

df_train = pd.get_dummies(df_train, columns=['states'], prefix='states')

df_test = pd.get_dummies(df_test, columns=['states'], prefix='states')
drop_column = ['id','date','zipcode', 'sqft_lot', 'sqft_living', 'sqft_basement'] #



df_train.drop(drop_column, axis=1, inplace = True)

df_test.drop(drop_column, axis=1, inplace = True)
X_train_data = df_train.drop('price', axis=1).values

X_train_label = df_train['price'].values

X_test_data = df_test.values



alg = XGBRegressor(

                     base_score=0.5,              

                     booster='gbtree',            

                     colsample_bytree=1,          

                     importance_type='gain',      

                     max_depth=3,                 

                     min_child_weight=1,          

                     n_estimators=10000,          

                     n_jobs=1,                   

                     scale_pos_weight=1,         

                     silent=True,               

                     gamma=0,                    

                     random_state=0,              

                     reg_alpha=0,                

                     reg_lambda=1,                   

                     subsample=1,                     

                     learning_rate=0.1           

                  )





test_acc = []

xgb_prediction = 0 

splits_cnt = 5

kf = KFold(n_splits=splits_cnt ,random_state=0 ,shuffle=True)

for fold_no,(train_index, test_index) in enumerate(kf.split(df_train)):

    

    train_data, test_data = X_train_data[train_index],  X_train_data[test_index]

    train_label, test_label = X_train_label[train_index], X_train_label[test_index]

    



    alg.fit(

              train_data, train_label

            , eval_set = [(train_data,train_label),(test_data, test_label)] 

            , eval_metric = 'rmse'        

            , verbose=0                   

            , early_stopping_rounds=500  

           )

    

    

    predict = alg.predict(test_data)

    predict_value = np.sqrt(mean_squared_error(np.exp(test_label),np.exp(predict))).round().astype(int)

    test_acc.append(predict_value)

    print(predict_value)

    

   

    predictions = alg.predict(X_test_data)

    xgb_prediction += np.exp(predictions)

    

print(sum(test_acc)/len(test_acc))   

xgb_final = xgb_prediction / splits_cnt    
submission = pd.read_csv('../input/sample_submission.csv')
submission['price'] = xgb_final
submission.head()
submission.to_csv('./xgb_final_to_verify.csv', index=False)