import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import json

import time

import pickle

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential

from keras.layers import Dense, Activation,Dropout, BatchNormalization

from sklearn.model_selection import KFold

import xgboost

from xgboost import plot_importance

import lightgbm as lgb

from sklearn.model_selection import  train_test_split

from sklearn.model_selection import GridSearchCV   #Performing grid search

from sklearn.model_selection import validation_curve

import gc

import os

print(os.listdir("../input"))
#Loading Data 

file_train = "../input/train_v2.csv"

file_test = "../input/test_v2.csv"

chunk_size = 10000



def load_data(file,chunk_size,nrows_load=None,test_data=False):

    df_res = pd.DataFrame()

    df_reader = pd.read_csv(file,

                            dtype={ 'date': str, 'fullVisitorId': str},

                            chunksize=10000)

    

    for cidx, df in enumerate(df_reader):

        df.reset_index(drop=True, inplace=True)   

        process_df(df,test_data)

        df_res = pd.concat([df_res,df ], axis=0).reset_index(drop=True)

        del df #free memory

        gc.collect()

        #print every 20 iterations

        if cidx % 20 == 0:

            print('{}: rows loaded: {}'.format(cidx, df_res.shape[0]))

        if nrows_load:

            if res.shape[0] >= nrows_load:

                break

    return df_res
#every column as key and the important features to extract from each column



def parse_json(x,s):

    res = json.loads(x)

    try:

        return res[s]

    except:

        return float('NaN') 



def process_df(df,test_data):

    #process date 

    df['days'] = df['date'].str[-2:]

    df['days'] = df['days'].astype(int)

    df['month'] = df['date'].str[-4:-2]

    df['month'] = df['month'].astype(int)

    df['year'] = df['date'].str[:4]

    df['year'] = df['year'].astype(int)



    #process json fields

    process_dict = {

        'totals':['transactionRevenue','newVisits','pageviews','hits'] ,

        'trafficSource':['campaign','source','medium'] ,

        'device':['browser'],

        'geoNetwork': ['country','city','continent','region','subContinent']

    }

 

    #add new columns from json in df

    for c,l in process_dict.items():

        for it in l:

            df[it] = df[c].apply(lambda x : parse_json(x,it))

    

    #process custom dimensions

    #df['customDimensions_index'] = df['customDimensions'].apply(lambda x : parse_json(x,'index'))

    #df['customDimensions_val'] = df['customDimensions'].apply(lambda x : parse_json(x,'value'))

    

    

    #labelencoding for continuous data

    cols = ['country','campaign','source','medium','continent','city','region','socialEngagementType','browser'

             ,'channelGrouping','subContinent','date']

    labelencoder_X=LabelEncoder()

    for c in cols:

        df.loc[:,c] = labelencoder_X.fit_transform(df.loc[:,c])

        

    

    #Dealing with missing values

    #transactionsRevenue and NewVisits:  nans ->  0

    df['transactionRevenue'].fillna(0,inplace=True)

    df['newVisits'].fillna(0,inplace=True)

    df['pageviews'].fillna(0,inplace=True)

    

    

    #Casting Str columns to int

    df['transactionRevenue'] = df['transactionRevenue'].astype('float32')

    df['newVisits']= df['newVisits'].astype('uint16')

    df['pageviews'] = df['pageviews'].astype('uint16')

    df['hits'] = df['hits'].astype('uint32')

    #df['index'] = df['index'].astype('uint32')

    

    #remove json field columns and some unwanted columns

    #(some removed for saving memory)

    

    rm_col = ['subContinent',

             'channelGrouping','date','continent','customDimensions','fullVisitorId']

    if test_data:

        rm_col = rm_col[:-1]

    df.drop(list(process_dict.keys()) + rm_col, axis=1,inplace=True)

    

#load and process

df = load_data(file_train,chunk_size)

df_test =load_data(file_test,chunk_size,test_data=True)
#percent not zero 

per = sum(df['transactionRevenue'] > 0) / len(df['transactionRevenue'])

print('Percentage of transactions greater than 0 :   {} %'.format(per*100))
#Setting up Data

X = df[df.columns[df.columns != 'transactionRevenue']]

Y = np.log1p(df['transactionRevenue'])

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.4, random_state=42)



X_test = df_test[df_test.columns[df_test.columns != 'transactionRevenue']]

Y_test = np.log1p(df_test['transactionRevenue'])



#Handling the imbalanced dataset and looking at how it effects training



#Sampling 

#Using all the 1% non zero transactions revenue with an equal num of rows from the zero results

indices_nonzero = np.where (y_train > 0)

indices_zero = np.where (y_train == 0)

num_nonzero = len(indices_nonzero[0])

num_zero = len(indices_zero[0])

print('Number of non-zero transactions revenues = ', num_nonzero)

print('Number of zero transactions revenues = ', num_zero)



#Creating a sample dataset containing 50% of rows with non-zero transactions 

all_indx = list(indices_zero[0][0:num_nonzero]) + list(indices_nonzero[0])

X_sample = X_train.iloc[all_indx,:]

y_sample = y_train.iloc[all_indx]

#split train validation 

X_train_sample, X_val_sample, y_train_sample, y_val_sample = train_test_split(X_sample, y_sample, test_size=0.45, random_state=42)

print('Sample X_Train shape ', X_train_sample.shape)
def create_model_xgboost(X_train,y_train,X_val=None,y_val=None):

    params = {'objective': 'reg:linear',

              'eval_metric': 'rmse',

              'min_child_weight':1.5,    

              'eta': 0.01,

              'max_depth': 4,

              'subsample': 0.7,

              'colsample_bytree': 0.6,

              'reg_alpha':1,

              'reg_lambda':0.45,

              'random_state': 42,

              'silent': True}

    

    xgb_train_data = xgboost.DMatrix(X_train, y_train)

    if not X_val is None:

        xgb_val_data = xgboost.DMatrix(X_val, y_val)

        evals=[(xgb_train_data, 'train'), (xgb_val_data, 'valid')]

    else:

        evals=[(xgb_train_data, 'train')]

    model = xgboost.train(params, xgb_train_data, 

                      num_boost_round=1200, 

                      evals= evals,

                      early_stopping_rounds=50, 

                      verbose_eval=300) 

    return model



#Model  definition

def create_model_nn(in_dim,layer_size=250):

    model = Sequential()

    model.add(Dense(layer_size,activation='relu',input_dim=in_dim))

    model.add(BatchNormalization())

    model.add(Dense(layer_size,activation='relu'))

    model.add(Dense(layer_size,activation='relu'))

    model.add(Dense(1, activation='linear'))

    adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=adam,loss='mse')    

    return model





def create_model_lgbm(X_train,y_train,X_val=None,y_val=None):

    dtrain = lgb.Dataset(X_train,label=y_train)

    dval = lgb.Dataset(X_val,label=y_val)



    param_up = {"objective" : "regression", "metric" : "rmse", 

               "max_depth": 2, "min_child_samples": 20, 

               "reg_alpha": 1.5, "reg_lambda": 1.5,

               "num_leaves" : 15, "learning_rate" : 0.1, 

               "subsample" : 1, "colsample_bytree" : 1, 

               "verbosity": -1,'data_random_seed':4}

    if not X_val is None:

        valid_sets = (dtrain,dval)

        valid_names = ['train','valid']

    else:

        valid_sets = (dtrain)

        valid_names = ['train']

    model = lgb.train(param_up,dtrain,num_boost_round=5000,valid_sets=valid_sets,valid_names=['train','valid'],verbose_eval=300,

                     early_stopping_rounds=50)

    return model


#Fitting and Training

def fit_train(x,y,X_val,y_val,layer_size=64,mod = 'nn'):    

    if mod =='nn':

        model = create_model_nn(x.shape[1],layer_size)

        history = model.fit(x, y, epochs=3, batch_size=128,validation_data=(X_val,y_val),verbose=1)

        print('Mean RMSE : ',np.sqrt(history.history['val_loss']).mean())

    elif mod =='xgboost':

         model = create_model_xgboost(x,y,X_val,y_val)

    elif mod =='lgbm':

        model = create_model_lgbm(x,y,X_val,y_val)

    else:

        raise Exception ('invalid model')

    return model





def calc_rmse(pred,y):

    diff =  pred - y

    RMSE = ((diff ** 2).mean()) ** .5

    print('RMSE : ',RMSE)



#Evaluation

def eval_set(model,x,y,mod='nn'):

        if mod == 'nn':

            p = model.predict(x,batch_size=64,verbose=1)

            pred = p[:,0]

        elif mod == 'xgboost':

            dx = xgboost.DMatrix(x)

            pred = model.predict(dx,ntree_limit=model.best_ntree_limit)

        elif mod == 'lgbm':

            pred = model.predict(x, num_iteration=model.best_iteration)

        else:

            raise Exception ('invalid model')

        calc_rmse(pred,y)
from keras import backend as K

import tensorflow as tf

from keras import optimizers

import keras as k





#Training

print('Training on training/val Dataset')

model_training = fit_train(X_train,y_train,X_val,y_val)

print('-'*40)

print('Training on Sample Dataset')

model_sample = fit_train(X_train_sample,y_train_sample,X_val_sample,y_val_sample,layer_size=500)

print('-'*40)

print('Training on Full training Dataset')

model_full = create_model_nn(X.shape[1])

model_full.fit(X, Y, epochs=3, batch_size=128,verbose=1)



print('-'*40)
#Training

print('Training on training/val Dataset')

model_training = fit_train(X_train,y_train,X_val,y_val,mod='xgboost')

print('-'*40)

print('Training on Sample Dataset')

model_sample = fit_train(X_train_sample,y_train_sample,X_val_sample,y_val_sample,layer_size=500,mod='xgboost')

print('-'*40)

print('Training on Full training Dataset')

model_full = create_model_xgboost(X, Y)



print('Full Training set -- Using Sample training model')

eval_set(model_sample,X_train,y_train,mod='xgboost')
#Training

print('Training on training/val Dataset')

model_training = fit_train(X_train,y_train,X_val,y_val,mod='lgbm')

print('-'*40)

print('Training on Sample Dataset')

model_sample = fit_train(X_train_sample,y_train_sample,X_val_sample,y_val_sample,layer_size=500,mod='lgbm')

print('-'*40)

print('Training on Full training Dataset')

model_full = create_model_lgbm(X, Y)



print('Full Training set -- Using Sample training model')

eval_set(model_sample,X_train,y_train,mod='lgbm')



print('-'*40)
#Baseline Predictions

df_test['predictions'] = model_full.predict(df_test.loc[:,df_test.columns[1:]],num_iteration=model_full.best_iteration)

df_test.loc[df_test['predictions']< 0 ,'predictions']= 0

#set up dataframe for submission

sub_df = pd.DataFrame({'fullVisitorId':df_test['fullVisitorId'] , 'PredictedLogRevenue': np.expm1(df_test['predictions'])})

sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()

sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]

sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])

sub_df.to_csv("baseline_lgb_submission.csv", index=False)