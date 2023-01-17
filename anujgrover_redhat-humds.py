# importing basic libraries

import sys

import pandas as pd



from pandas import Series,DataFrame

import xgboost as xgb



# importing libraries we need for analysis and visualizations



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

#import datetime as dt

%matplotlib inline



# File 1 People.CSV file

# Description -- The people file contains all of the unique people (and the corresponding characteristics) that have 

#performed activities over time Each row in the people file represents a unique person. 

#Each person has a unique people_id.



people=pd.read_csv('../input/people.csv', 

                       dtype={'people_id' : np.str,

                              'activity_id' : np.str,

                              'char_38' : np.int8

                             }, parse_dates=['date']

                  )



act_train=pd.read_csv('../input/act_train.csv', 

                       dtype={'people_id' : np.str,

                              'activity_id' : np.str,

                              'outcome' : np.int8

                             }, parse_dates=['date']

                  )



act_test=pd.read_csv('../input/act_test.csv', 

                       dtype={'people_id' : np.str,

                              'activity_id' : np.str

                             }, parse_dates=['date']

                  )



#people.head()

#People Data Cleanup

#coverting boolean and object type to numerical for ski-kit to work properly



def data_xform(ds):

    x= ds

    for i in list(x.columns):

        if i not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:

            if x[i].dtype == 'object':

                x[i].fillna('type -99', inplace=True)

                x[i] = x[i].map(lambda x: x.split(' ')[1]).astype(np.int32)

            elif x[i].dtype == 'bool' :

                x[i] = x[i].astype(np.int8)

         

    x['year'] = x['date'].dt.year

    x['month'] = x['date'].dt.month

    x['day'] = x['date'].dt.day

    x.drop('date',axis=1,inplace=True)

    return x





people = data_xform(people)

people.head()



#people.head()

#people.info()



act_train = data_xform(act_train)

act_test = data_xform(act_test)

ds_train = pd.merge(act_train, people, on = 'people_id', how ='inner').sort_values(['people_id'], ascending=[1])

ds_test = pd.merge(act_test, people, on = 'people_id', how ='left').sort_values(['people_id'], ascending=[1])



x = ds_train.drop(['activity_id', 'people_id', 'outcome'], axis=1)

a = list(set(x.columns.values))

print(a)



x_train = ds_train[a]

y_train = ds_train['outcome']

x_test = ds_test[a]

#print(x_test)



x_train = x_train[a].as_matrix()



import xgboost as xgb



#ds_test.set_index('activity_id',inplace='True')

#ds_train.set_index('activity_id',inplace='True')



dtrain = xgb.DMatrix(x_train,y_train)

#print (dtrain)



dtest = xgb.DMatrix(x_test)



params = {"objective": "binary:logistic",

          "booster" : "gbtree",

          "eval_metric": "auc",

          "eta": 0.8,

          "tree_method": 'exact',

          "max_depth": 5,

          "subsample": 0.8,

          "colsample_bytree": 0.8,

          "silent": 0 }

num_boost_round = 110

early_stopping_rounds = 10



#Training Started 



#Training Started 

boost = xgb.train(params, dtrain, num_boost_round,early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

x_predict=boost.predict(dtest)

#y_predict

output = pd.DataFrame({ 'activity_id' : ds_test['activity_id'], 'outcome': x_predict })

output.head()