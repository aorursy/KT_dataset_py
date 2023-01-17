# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import xgboost as xgb

import pandas as pd

from sklearn import preprocessing, pipeline, metrics, grid_search

import time

import random

import numpy as np

import matplotlib.pyplot as plt

import math

from sklearn.metrics import mean_absolute_error



from scipy import sparse

%matplotlib inline
def logregobj(labels, preds):

    con = 2

    x =preds-labels

    grad =con*x / (np.abs(x)+con)

    hess =con**2 / (np.abs(x)+con)**2

    return grad, hess 



def log_mae(labels,preds,lift=200):

    return mean_absolute_error(np.exp(labels)-lift, np.exp(preds)-lift)



log_mae_scorer = metrics.make_scorer(log_mae, greater_is_better = False)



def search_model(train_x, train_y, est, param_grid, n_jobs, cv, refit=False):

##Grid Search for the best model

    model = grid_search.GridSearchCV(estimator  = est,

                                     param_grid = param_grid,

                                     scoring    = log_mae_scorer,

                                     verbose    = 10,

                                     n_jobs  = n_jobs,

                                     iid        = True,

                                     refit    = refit,

                                     cv      = cv)

    # Fit Grid Search Model

    model.fit(train_x, train_y)

    print("Best score: %0.3f" % model.best_score_)

    print("Best parameters set:", model.best_params_)

    print("Scores:", model.grid_scores_)

    return model







def xg_eval_mae(yhat, dtrain, lift=200):

    y = dtrain.get_label()

    return 'mae', mean_absolute_error(np.exp(y)-lift, np.exp(yhat)-lift)



def xgb_logregobj(preds, dtrain):

    con = 2

    labels = dtrain.get_label()

    x =preds-labels

    grad =con*x / (np.abs(x)+con)

    hess =con**2 / (np.abs(x)+con)**2

    return grad, hess





def search_model_mae (train_x, train_y, est, param_grid, n_jobs, cv, refit=False):

##Grid Search for the best model

    model = grid_search.GridSearchCV(estimator  = est,

                                     param_grid = param_grid,

                                     scoring    = 'neg_mean_absolute_error',

                                     verbose    = 10,

                                     n_jobs  = n_jobs,

                                     iid        = True,

                                     refit    = refit,

                                     cv      = cv)

    # Fit Grid Search Model

    model.fit(train_x, train_y)

    print("Best score: %0.3f" % model.best_score_)

    print("Best parameters set:", model.best_params_)

    print("Scores:", model.grid_scores_)

    return model
# Load data

start = time.time() 

train_data = pd.read_csv('../input/train.csv')

train_size=train_data.shape[0]

print ("Loading train data finished in %0.3fs" % (time.time() - start))        



test_data = pd.read_csv('../input/test.csv')

print ("Loading test data finished in %0.3fs" % (time.time() - start))        
full_data=pd.concat([train_data

                       ,test_data])

del( train_data, test_data)

print ("Full Data set created.")
data_types = full_data.dtypes  

cat_cols = list(data_types[data_types=='object'].index)

num_cols = list(data_types[data_types=='int64'].index) + list(data_types[data_types=='float64'].index)



id_col = 'id'

target_col = 'loss'

num_cols.remove('id')

num_cols.remove('loss')



print ("Categorical features:", cat_cols)

print ( "Numerica features:", num_cols)

print ( "ID: %s, target: %s" %( id_col, target_col))