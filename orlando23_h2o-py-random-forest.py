# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os



import h2o

from h2o.estimators.random_forest import H2ORandomForestEstimator



import numpy as np

import pandas as pd



os.environ['http_proxy'] = ''

os.environ['https_proxy'] = ''

os.environ['NO_PROXY'] = 'localhost'



def transform(df_):

    """

    transform Date to datetime type.

    """

    df_['Date']=pd.to_datetime(df_['Date'])

    df_['month']=df_['Date'].dt.month

    df_['year']=df_['Date'].dt.year

    df_['Store']=pd.to_numeric(df_['Store'])

    return df_



store=pd.read_csv("../input/store.csv")

train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

# filter sales invalid column

train=train[train['Sales'] > 0]

# merge train and test with store

train=pd.merge(train, store, on=['Store'])

test=pd.merge(test, store, on=['Store'])

# transform

train=transform(train)

test=transform(test)

# train convert to logSales from log(Sales)

train['logSales']=pd.to_numeric(np.log(train['Sales']))

test['logSales']=0



# initialization of h2o

h2o.init(nthreads=-1, max_mem_size = "8G")

train_hf = h2o.H2OFrame(train)

test_hf = h2o.H2OFrame(test)

rf_v1_model = H2ORandomForestEstimator(model_id="rf_covType_v1", ntrees=200, stopping_rounds=2, max_depth = 30, nbins_cats = 1115, score_each_iteration=True, seed=1000000)



# Training prepare for model

covtype_X=[col for col in train_hf.columns if col not in ["Id","Date","Sales","logSales","Customers"]]

covtype_y=train_hf.columns[-1]

rf_v1_model.train(x=covtype_X, y=covtype_y, training_frame=train_hf)

test_result_hf = rf_v1_model.predict(test_hf)

test_result_df = test_result_hf.as_data_frame()

test_result_df['predict']=np.exp(test_result_df['predict'])

test_result_df.rename(columns={'predict': 'Sales'}, inplace=True)

test_result_df.insert(loc=0, column='Id', value=test['Id'])

test_result_df.set_index('Id')

test_result_df.to_csv('python_h2o_rf.csv', header=True, index=False)