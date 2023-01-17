# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import datetime as dt

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.preprocessing import StandardScaler

import os

print(os.listdir("../input"))

pd.options.mode.chained_assignment = None

pd.set_option('display.max_columns', 500)

from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold 

# Any results you write to the current directory are saved as output.


df_train_f1=pd.read_csv("../input/fraud-data2/traindata2.csv", sep=',', lineterminator='\n',chunksize=500000)





is_frd =  df_train['isFraud']==1

df_train_1=df_train[is_frd]



df_train_1_1=df_train_1

#df_train_1_1=df_train_1[:15]

#df_train_1_2=df_train_1[15:37] # this set will be added to  test set





print(df_train_1.shape)

print(df_train_1_1.shape)

print(df_train_1_2.shape)





is_not_frd=df_train['isFraud']==0

df_train_0=df_train[is_not_frd]

print(df_train_0.shape)





df_train_f1=pd.concat([df_train_0,df_train_1_1],ignore_index=True)

df_train_f1 = shuffle(df_train_f1)

print(df_train_f1.shape)





"""

Index(['Id', 'step', 'action', 'amount', 'nameOrig', 'oldBalanceOrig',

       'newBalanceOrig', 'nameDest', 'oldBalanceDest', 'newBalanceDest',

       'isFraud', 'isFlaggedFraud\r'],

      dtype='object')

"""


chunk_list = [] 

#Cross-validation 

params ={

    # Parameters that we are going to tune.

    'n_estimators':10,

    'max_depth': 2, #Result of tuning with CV

    'eta':0.03, #Result of tuning with CV

    #'subsample': 1, #Result of tuning with CV

    #'colsample_bytree': 0.8, #Result of tuning with CV

    # Other parameters

    #'objective':'reg:linear',

    #'eval_metric':'rmse',

    #'silent': 1

}





model = xgb.XGBClassifier(params=params)

chunk_idx = 0

# Each chunk is in df format

for chunk in df_train_f1: 

    chunk.rename(columns={'isFlaggedFraud\r':'isFlaggedFraud'}, inplace=True)

    

    

    

    # TURN CHARACTER VALUES INTO NUMERICS

    lb_make = LabelEncoder()

    chunk["nameOrig_num"] = lb_make.fit_transform(chunk["nameOrig"])

    #print(df_train_f1[["nameOrig_num","nameOrig"]])

    chunk["nameDest_num"] = lb_make.fit_transform(chunk["nameDest"])

    chunk["action_num"] = lb_make.fit_transform(chunk["action"])

    

    grouped = chunk['amount'].groupby(chunk['nameOrig_num'])

    grouped = grouped.mean()

    df_g1=pd.DataFrame(grouped)

    #print(df_g1.head())

    grouped = chunk['amount'].groupby(chunk['nameOrig_num'])

    grouped = grouped.std(ddof=0)

    df_g2=pd.DataFrame(grouped)



    grouped = chunk['nameDest_num'].groupby(chunk['nameOrig_num'])

    grouped = grouped.count()

    df_g3=pd.DataFrame(grouped)





    chunk = chunk.merge(df_g1, on=['nameOrig_num'],suffixes=("", "_mean"))

    chunk = chunk.merge(df_g2, on=['nameOrig_num'],suffixes=("", "_std"))

    chunk = chunk.merge(df_g3, on=['nameOrig_num'],suffixes=("", "_count"))

    

    if chunk_idx > 0:

        df_train_4.append(chunk)

        #df_train_4=df_train_2

    else:

        df_train_4=chunk

        

    lister_4=['step', 'amount', 'oldBalanceOrig','newBalanceOrig', 'oldBalanceDest', 'newBalanceDest',

    'isFlaggedFraud', 'nameOrig_num', 'nameDest_num','action_num', 'amount_mean', 'amount_std',

    'nameDest_num_count']

    lister_5=['isFraud']

    

    

    #print(df_train_4.isFraud.unique())

    



    data=df_train_4[lister_4]

    target=df_train_4[lister_5]

    print("before K FOLD ",target.isFraud.unique())

    

    K = 2

    kf = KFold(n_splits=K, shuffle=True, random_state=42)



    for train_index, val_index in kf.split(data, target):

        

        # split data

        X_train, X_test = data.iloc[train_index], data.iloc[val_index]

        y_train, y_test = target.iloc[train_index], target.iloc[val_index]

        print(y_test.isFraud.unique())

        if chunk_idx > 0: # not load in first run

                model.fit(X_train, y_train, xgb_model='model_1.model')

                model.save_model('model_1.model')

        else:

                model.fit(X_train, y_train)

                model.save_model('model_1.model')

                

    chunk_idx = chunk_idx + 1

    rmse = sqrt(mean_squared_error(y_test, model.predict(X_test)))

    print("RMSE RESULT ",rmse)

    # evaluate predictions

    accuracy = accuracy_score(y_test, model.predict(X_test))

    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    

    predictions=model.predict(X_test)

    print(predictions.shape)

    print(y_test.shape)

    #print(y_test)

    print(predictions)

    my_submission = pd.DataFrame( { 'key': y_test.isFraud,'fare_amount': predictions } )

    """"

    """

    X_train, X_test, y_train, y_test= train_test_split(df_train_4[lister_4],df_train_4[lister_5], test_size=0.3,

    random_state=42)

    

    print(y_test.isFraud.unique())

    

    if chunk_idx > 0: # not load in first run

        model.fit(X_train, y_train, xgb_model='model_1.model')

        model.save_model('model_1.model')

    else:

        model.fit(X_train, y_train)

        model.save_model('model_1.model')

    chunk_idx = chunk_idx + 1

    rmse = sqrt(mean_squared_error(y_test, model.predict(X_test)))

    print("RMSE RESULT ",rmse)

    # evaluate predictions

    accuracy = accuracy_score(y_test, model.predict(X_test))

    print("Accuracy: %.2f%%" % (accuracy * 100.0))





my_submission.to_csv('sample_submission.csv', index=False)

print("Writing complete")
