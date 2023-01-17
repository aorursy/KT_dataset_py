import numpy as np

import xgboost as xgb

import pandas as pd

import math

import os

import sys

%matplotlib inline



from sklearn.cross_validation import train_test_split

from ml_metrics import rmsle
def getVariables(value=1000):

    for var, obj in globals().items():

        try:

            if(sys.getsizeof(obj) > value and not var.startswith("_")):

                    print ("{0:30} {1:5}".format(var, sys.getsizeof(obj)))

        except:

            continue
def evalerror(preds, dtrain):



    labels = dtrain.get_label()

    assert len(preds) == len(labels)

    labels = labels.tolist()

    preds = preds.tolist()

    terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0,preds[i]) + 1)) ** 2.0 for i,pred in enumerate(labels)]

    return 'error', (sum(terms_to_sum) * (1.0/len(preds))) ** 0.5
print ('Loading Test...')

dtype_test = {'id':np.uint32,

              'Semana': np.uint8, 

              'Agencia_ID': np.uint16, 

              'Canal_ID': np.uint8,

              'Ruta_SAK': np.uint16, 

              'Cliente_ID': np.uint32, 

              'Producto_ID': np.uint16}



%time test = pd.read_csv('../input/test.csv', usecols=dtype_test.keys(), dtype=dtype_test)

test.head()
test.shape
dtype = {'Semana': np.uint8, 

         'Agencia_ID': np.uint16, 

         'Canal_ID': np.uint8,

         'Ruta_SAK': np.uint16, 

         'Cliente_ID': np.uint32, 

         'Producto_ID': np.uint16,

         'Demanda_uni_equil': np.uint16}



filename='../input/train.csv'





%time train = pd.read_csv(filename, usecols=dtype.keys(), dtype=dtype, warn_bad_lines= True,engine='c')

train.head()
train = train[train["Semana"]>8]

print ('Training_Shape:', train.shape)
ids = test['id']

test = test.drop(['id'],axis = 1)



y = train['Demanda_uni_equil']

X = train[test.columns.values]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1729)

del(train)

print ('Division_Set_Shapes:', X.shape, y.shape)

print ('Validation_Set_Shapes:', X_train.shape, X_test.shape)

del(X)

del(y)
params = {}

params['objective'] = "reg:linear"

params['eta'] = 0.1

params['max_depth'] = 5

params['subsample'] = 0.8

params['colsample_bytree'] = 0.6

params['silent'] = True

#params['nthread']= 4

params['booster'] = "gbtree"





test_preds = np.zeros(test.shape[0])

xg_train = xgb.DMatrix(X_train, label=y_train)

del(X_train)

del(y_train)

xg_test = xgb.DMatrix(X_test)

del(X_test)

watchlist = [(xg_train, 'train')]
num_rounds = 20

%time xgclassifier = xgb.train(params, xg_train, num_rounds, watchlist, feval = evalerror, early_stopping_rounds= 20, verbose_eval = 10)

del(xg_train)
preds = xgclassifier.predict(xg_test, ntree_limit=xgclassifier.best_iteration)

print ('RMSLE Score:', rmsle(y_test, preds)) 
print ('RMSLE Score:', rmsle(y_test, preds)) 

del(preds)

del(y_test)