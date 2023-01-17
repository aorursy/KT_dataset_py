# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import os

import time

import numpy as np

import pandas as pd

from seaborn import countplot,lineplot, barplot

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, confusion_matrix



from bayes_opt import BayesianOptimization

import lightgbm as lgb

import xgboost as xgb

import catboost as cb
from IPython.display import HTML

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
tr = pd.read_csv('../input/X_train.csv')

te = pd.read_csv('../input/X_test.csv')

target = pd.read_csv('../input/y_train.csv')

ss = pd.read_csv('../input/sample_submission.csv')
def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z



def fe(actual):

    new = pd.DataFrame()

    actual['total_angular_velocity'] = (actual['angular_velocity_X'] ** 2 + actual['angular_velocity_Y'] ** 2 + actual['angular_velocity_Z'] ** 2) ** 0.5

    actual['total_linear_acceleration'] = (actual['linear_acceleration_X'] ** 2 + actual['linear_acceleration_Y'] ** 2 + actual['linear_acceleration_Z'] ** 2) ** 0.5

    

    actual['acc_vs_vel'] = actual['total_linear_acceleration'] / actual['total_angular_velocity']

    

    x, y, z, w = actual['orientation_X'].tolist(), actual['orientation_Y'].tolist(), actual['orientation_Z'].tolist(), actual['orientation_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    

    actual['total_angle'] = (actual['euler_x'] ** 2 + actual['euler_y'] ** 2 + actual['euler_z'] ** 2) ** 5

    actual['angle_vs_acc'] = actual['total_angle'] / actual['total_linear_acceleration']

    actual['angle_vs_vel'] = actual['total_angle'] / actual['total_angular_velocity']

    

    def f1(x):

        return np.mean(np.diff(np.abs(np.diff(x))))

    

    def f2(x):

        return np.mean(np.abs(np.diff(x)))

    

    for col in actual.columns:

        if col in ['row_id', 'series_id', 'measurement_number']:

            continue

        new[col + '_mean'] = actual.groupby(['series_id'])[col].mean()

        new[col + '_min'] = actual.groupby(['series_id'])[col].min()

        new[col + '_max'] = actual.groupby(['series_id'])[col].max()

        new[col + '_std'] = actual.groupby(['series_id'])[col].std()

        new[col + '_max_to_min'] = new[col + '_max'] / new[col + '_min']

        

        # Change. 1st order.

        new[col + '_mean_abs_change'] = actual.groupby('series_id')[col].apply(f2)

        

        # Change of Change. 2nd order.

        new[col + '_mean_change_of_abs_change'] = actual.groupby('series_id')[col].apply(f1)

        

        new[col + '_abs_max'] = actual.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

        new[col + '_abs_min'] = actual.groupby('series_id')[col].apply(lambda x: np.min(np.abs(x)))



    return new
tr = fe(tr)

te = fe(te)
train_labels= target['surface']
tr.fillna(0, inplace = True)

te.fillna(0, inplace = True)

tr.replace(-np.inf, 0, inplace = True)

tr.replace(np.inf, 0, inplace = True)

te.replace(-np.inf, 0, inplace = True)

te.replace(np.inf, 0, inplace = True)
le = LabelEncoder()

train_label_encoded = le.fit_transform(train_labels)
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(tr,train_label_encoded,test_size = 0.33 ,random_state = 50)
train_set = lgb.Dataset(train_features, label = train_labels)

test_set = lgb.Dataset(test_features, label = test_labels)
import csv

from hyperopt import STATUS_OK

from timeit import default_timer as timer

from hyperopt import hp

from hyperopt.pyll.stochastic import sample
space = {

    'boosting_type': hp.choice('boosting_type', 

                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 

                                             {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},

                                             {'boosting_type': 'goss', 'subsample': 1.0}]),

    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),

    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),

    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),

    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),

    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),

    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),

    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),

    'is_unbalance': hp.choice('is_unbalance', [True, False]),

}

x = sample(space)



# Conditional logic to assign top-level keys

subsample = x['boosting_type'].get('subsample', 1.0)

x['boosting_type'] = x['boosting_type']['boosting_type']

x['subsample'] = subsample

from hyperopt import tpe



# Create the algorithm

tpe_algorithm = tpe.suggest





from hyperopt import Trials



# Record results

trials = Trials()


from hyperopt import fmin





# Global variable

global  ITERATION



ITERATION = 0



# Run optimization

# best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,

#             max_evals = MAX_EVALS)



# best



def objective(hyperparameters):

    """Objective function for Gradient Boosting Machine Hyperparameter Optimization.

       Writes a new line to `outfile` on every iteration"""

    

    # Keep track of evals

    global ITERATION

    

    ITERATION += 1

    

    # Using early stopping to find number of trees trained

    if 'n_estimators' in hyperparameters:

        del hyperparameters['n_estimators']

    

    # Retrieve the subsample

    subsample = hyperparameters['boosting_type'].get('subsample', 1.0)

    

    # Extract the boosting type and subsample to top level keys

    hyperparameters['boosting_type'] = hyperparameters['boosting_type']['boosting_type']

    hyperparameters['subsample'] = subsample

    

    # Make sure parameters that need to be integers are integers

    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:

        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])



    start = timer()

    

    # Perform n_folds cross validation

    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round = 10000, nfold = N_FOLDS, 

                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)



    run_time = timer() - start

    

    # Extract the best score

    best_score = cv_results['auc-mean'][-1]

    

    # Loss must be minimized

    loss = 1 - best_score

    

    # Boosting rounds that returned the highest cv score

    n_estimators = len(cv_results['auc-mean'])

    

    # Add the number of estimators to the hyperparameters

    hyperparameters['n_estimators'] = n_estimators



    # Write to the csv file ('a' means append)

#     of_connection = open(OUT_FILE, 'a')

#     writer = csv.writer(of_connection)

#     writer.writerow([loss, hyperparameters, ITERATION, run_time, best_score])

#     of_connection.close()



    # Dictionary with information for evaluation

    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,

            'train_time': run_time, 'status': STATUS_OK}
N_FOLDS = 5

MAX_EVALS = 5

# best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,

#             max_evals = MAX_EVALS)