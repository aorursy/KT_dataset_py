# Required libraries

# We will try several Machine Learning platforms

from __future__ import print_function

from builtins import str

from builtins import range



import os

import sys

import tarfile



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from io import BytesIO



import bson

import json 

import skimage



import matplotlib.pyplot as plt

import keras

import tensorflow as tf



from sklearn import *

from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

import time

import datetime as dt





import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import log_loss



# Config the matplotlib backend as plotting inline in IPython

%matplotlib inline





from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



print("tf.__version__ : ", tf.__version__)

print("python --version : ", sys.version)

PyVersion = sys.version
# Read data

train = pd.read_json("../input/train.json")

#test = pd.read_json("test.json")

train.inc_angle = train.inc_angle.replace('na', 0)

train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

print("Total number of images :", len(train))

train.head(0)

print("done!")

train[:7]
# Test data

test = pd.read_json('../input/test.json')

test['inc_angle'] = pd.to_numeric(test['inc_angle'],errors='coerce')

print("Total number of images :", len(test))

test.head(0)

test[:7]
# Train data

def get_stats(train,label=1):

    train['max'+str(label)] = [np.max(np.array(x)) for x in train['band_'+str(label)] ]

    train['maxpos'+str(label)] = [np.argmax(np.array(x)) for x in train['band_'+str(label)] ]

    train['min'+str(label)] = [np.min(np.array(x)) for x in train['band_'+str(label)] ]

    train['minpos'+str(label)] = [np.argmin(np.array(x)) for x in train['band_'+str(label)] ]

    train['med'+str(label)] = [np.median(np.array(x)) for x in train['band_'+str(label)] ]

    train['std'+str(label)] = [np.std(np.array(x)) for x in train['band_'+str(label)] ]

    train['mean'+str(label)] = [np.mean(np.array(x)) for x in train['band_'+str(label)] ]

    train['p25_'+str(label)] = [np.sort(np.array(x))[int(0.25*75*75)] for x in train['band_'+str(label)] ]

    train['p75_'+str(label)] = [np.sort(np.array(x))[int(0.75*75*75)] for x in train['band_'+str(label)] ]

    train['mid50_'+str(label)] = train['p75_'+str(label)]-train['p25_'+str(label)]



    return train

train = get_stats(train,1)

train = get_stats(train,2)
train.head(2)
col1 = ['min1','max1','std1','med1','mean1','mid50_1']

col2 = ['min2','max2','std2','med2','mean2','mid50_2']

col = [c for c in train.columns if c not in ['id','is_iceberg', 'band_1', 'band_2']]

#col = [c for c in train.columns if c not in ['id','is_iceberg', 'inc_angle', 'band_1', 'band_2']]
len(col)
# We could try several augmentation methods on the data to see the effect

# Standardize values to 0 mean and unit standard deviation

min_max_scaler = preprocessing.MinMaxScaler()

train_minmax = min_max_scaler.fit_transform(train[col])



# DATA SPLITING

X_train, X_test, y_train, y_test = train_test_split(train[col], train['is_iceberg'], test_size=0.25, random_state=42)

#X_train, X_test, y_train, y_test = train_test_split(train_minmax, train['is_iceberg'], test_size=0.25, random_state=42)



X_train = X_train.values.astype(np.float32)

X_test = X_test.values.astype(np.float32)

y_train = y_train.values.astype(np.int)

y_test = y_test.values.astype(np.int)

#xtest = test[col].values.astype(np.float32)





n_features = X_train.shape[1]



n_classes = len(np.unique(y_train))



print("n_features : {}\nn_classes : {}\nX_train.shape : {}".format(n_features, n_classes, X_train.shape))
X_train.shape
X_train[1]
y_train.shape


print('Start training...')

# train

gbm = lgb.LGBMClassifier(objective='binary',

                        num_leaves=31,

                        learning_rate=0.05,

                        n_estimators=20)

gbm.fit(X_train, y_train,

        eval_set=[(X_test, y_test)],

        eval_metric='binary_logloss',

        early_stopping_rounds=100)



print('Start predicting...')

# predict

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

# eval

print('The log_loss of prediction is:', log_loss(y_test, y_pred))



# feature importances

print('\nNumber of features :', len(list(gbm.feature_importances_)))

print('Features :', col)

print('Importances :', list(gbm.feature_importances_))

print('\nFeature importances :', dict(zip(col,list(gbm.feature_importances_))))





### ### Hyperparameter Optimization ##############

# other scikit-learn modules

estimator = lgb.LGBMClassifier(num_leaves=31)



# The parameters used are in comment below, it will take too long time to run them here

param_grid = {

    'learning_rate': [0.1],

    'n_estimators': [100, 500],

    'num_leaves': [20, 31],

    'min_data_in_leaf': [5, 10],

    'reg_alpha': [0],

    'reg_lambda': [1e-6], 

    'bagging_fraction': [0.8, 0.9],

    'min_child_samples': [10, 20],

    'min_child_weight': [1e-6], 

    'max_bin': [256]

}



gbm = GridSearchCV(estimator, param_grid)



gbm.fit(X_train, y_train)



print('\n\nBest parameters found by grid search are:', gbm.best_params_)



'''

param_grid = {

    'learning_rate': [0.01, 0.1, 0.05, 0.07, 1],

    'n_estimators': [20, 40, 100, 500],

    'num_leaves': [20, 31, 50, 127],

    'min_data_in_leaf': [5, 10, 20, 50, 100],

    'reg_alpha': [0, 1e-3, 1e-6],

    'reg_lambda': [0, 1e-3, 1e-6], 

    'bagging_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],

    'min_child_samples': [10, 20, 30],

    'min_child_weight': [5, 1e-3, 1e-6], 

    'max_bin': [255, 256]

}

'''

#
# FEATURES TUNING IF NECESSARY

## For example, we could remove least important features if required

## And also use the best parameters provided by the grid search Cross Validation
# Here I reuse the same previous splits instead of recreate a new one.

X_train = pd.DataFrame(X_train, columns=col)

X_test = pd.DataFrame(X_test, columns=col)
X_train.head(2)
X_test.head(2)
# I decided here to delete the features with lower importance (7 and 8 values) to see how that could improve the result

# Using the Feature importances Dictionary

new_cols = [c for c in train.columns if c not in ['id','is_iceberg', 'band_1', 'band_2', 'p75_2', 'mean2', 'minpos2']]

len(new_cols)
X_train_new = X_train[new_cols]

X_test_new = X_test[new_cols]

X_train_new.shape
# Come back into arrays for training

X_train_new = X_train_new.values.astype(np.float32)

X_test_new = X_test_new.values.astype(np.float32)
# TRAINING
# specify your configurations as a dict

# Use the best parameters provided by the grid search Cross Validation

params = {"objective": "binary",

          #"sigmoid":1.0,

          "task": "train",

          "boosting_type": "gbdt",

          "learning_rate": 0.1,

          "num_leaves": 20, # 31

          "max_bin": 256,

          "min_data_in_leaf": 5, # Problem  2000

          "feature_fraction": 0.6, # 0.6

          "verbosity": 0,

          "seed": 0,

          "drop_rate": 0.1, # 0.1

          "is_unbalance": False,

          "max_drop": 50,

          "min_child_samples": 10,

          "min_child_weight": 1e-06, # 5

          "min_split_gain": 0,

          "colsample_bytree": 0.6343275033,

          "max_depth": 8, # 8

          "n_estimators": 500, # 500

          "nthread": -1,

          "reg_alpha": 0,

          "reg_lambda": 1e-06,# 1

          "silent": True,

          "subsample_for_bin": 50000, # 50000

          "subsample_freq": 1, # 1

          #"min_data":1,

          #"min_data_in_bin":1,

          'metric': {'binary_logloss'},

          'bagging_fraction': 0.8,

          'bagging_freq': 5,

          #'num_iterations':1000,

          "subsample": 0.733

          }

y_pred[:12]
# create dataset for lightgbm

lgb_train = lgb.Dataset(X_train_new, y_train)

lgb_eval = lgb.Dataset(X_test_new, y_test, reference=lgb_train)





print('Start training...')

# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=2000,

                valid_sets=lgb_eval,

                early_stopping_rounds=100)



print('Save model...')

# save model to file

gbm.save_model('model.txt')



print('Start predicting...')

# predict

y_pred = gbm.predict(X_test_new, num_iteration=gbm.best_iteration)

# eval

#print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

print('The log_loss of prediction is:', log_loss(y_test, y_pred))
# Lets ensure first that the test set is under the same preprocessing as the train set (to reproduce the trainer performance).

test = get_stats(test,1)

test = get_stats(test,2)
# xtest = min_max_scaler.fit_transform(test[new_cols])

xtest = test[new_cols]

preds = gbm.predict(xtest, num_iteration=gbm.best_iteration)
preds
submission = pd.DataFrame({'id': test["id"], 'is_iceberg': preds})

submission.head(10)
submission.to_csv("./LightGBM_CV_submission.csv", index=False)
from IPython.display import FileLink

#%cd $LESSON_HOME_DIR

FileLink('LightGBM_CV_submission.csv')