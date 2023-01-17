# This line is needed for python 2.7 ; probably not for python 3

from __future__ import print_function



import numpy as np

import pandas as pd

import gc

import warnings



from bayes_opt import BayesianOptimization



from sklearn.cross_validation import cross_val_score, StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score

from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb

import contextlib
#@contextlib.contextmanager

#def capture():

#    import sys

#    from cStringIO import StringIO

#    olderr, oldout = sys.stderr, sys.stdout

#    try:

#        out=[StringIO(), StringIO()]

#        sys.stderr,sys.stdout = out

#        yield out

#    finally:

#        sys.stderr,sys.stdout = olderr,oldout

#        out[0] = out[0].getvalue().splitlines()

#        out[1] = out[1].getvalue().splitlines()
def scale_data(X, scaler=None):

    if not scaler:

        scaler = MinMaxScaler(feature_range=(-1, 1))

        scaler.fit(X)

    X = scaler.transform(X)

    return X, scaler
DATA_TRAIN_PATH = '../input/train.csv'

DATA_TEST_PATH = '../input/test.csv'



def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH):

    train_loader = pd.read_csv(path_train, dtype={'target': np.int8, 'id': np.int32})

    train = train_loader.drop(['target', 'id'], axis=1)

    train_labels = train_loader['target'].values

    train_ids = train_loader['id'].values

    print('\n Shape of raw train data:', train.shape)



    test_loader = pd.read_csv(path_test, dtype={'id': np.int32})

    test = test_loader.drop(['id'], axis=1)

    test_ids = test_loader['id'].values

    print(' Shape of raw test data:', test.shape)



    return train, train_labels, test, train_ids, test_ids
# Comment out any parameter you don't want to test

def XGB_CV(

          max_depth,

          gamma,

          min_child_weight,

          max_delta_step,

          subsample,

          colsample_bytree

         ):



    global AUCbest

    global ITERbest



#

# Define all XGboost parameters

#



    paramt = {

              'booster' : 'gbtree',

              'max_depth' : int(max_depth),

              'gamma' : gamma,

              'eta' : 0.1,

              'objective' : 'binary:logistic',

              'nthread' : 4,

              'silent' : True,

              'eval_metric': 'auc',

              'subsample' : max(min(subsample, 1), 0),

              'colsample_bytree' : max(min(colsample_bytree, 1), 0),

              'min_child_weight' : min_child_weight,

              'max_delta_step' : int(max_delta_step),

              'seed' : 1001

              }



    folds = 5

    cv_score = 0



    print("\n Search parameters (%d-fold validation):\n %s" % (folds, paramt), file=log_file )

    log_file.flush()



    xgbc = xgb.cv(

                    paramt,

                    dtrain,

                    num_boost_round = 20000,

                    stratified = True,

                    nfold = folds,

#                    verbose_eval = 10,

                    early_stopping_rounds = 100,

                    metrics = 'auc',

                    show_stdv = True

               )



# This line would have been on top of this section

#    with capture() as result:



# After xgb.cv is done, this section puts its output into log file. Train and validation scores 

# are also extracted in this section. Note the "diff" part in the printout below, which is the 

# difference between the two scores. Large diff values may indicate that a particular set of 

# parameters is overfitting, especially if you check the CV portion of it in the log file and find 

# out that train scores were improving much faster than validation scores.



#    print('', file=log_file)

#    for line in result[1]:

#        print(line, file=log_file)

#    log_file.flush()



    val_score = xgbc['test-auc-mean'].iloc[-1]

    train_score = xgbc['train-auc-mean'].iloc[-1]

    print(' Stopped after %d iterations with train-auc = %f val-auc = %f ( diff = %f ) train-gini = %f val-gini = %f' % ( len(xgbc), train_score, val_score, (train_score - val_score), (train_score*2-1),

(val_score*2-1)) )

    if ( val_score > AUCbest ):

        AUCbest = val_score

        ITERbest = len(xgbc)



    return (val_score*2) - 1
# Define the log file. If you repeat this run, new output will be added to it

log_file = open('Porto-AUC-5fold-XGB-run-01-v1-full.log', 'a')

AUCbest = -1.

ITERbest = 0



# Load data set and target values

train, target, test, tr_ids, te_ids = load_data()

n_train = train.shape[0]

train_test = pd.concat((train, test)).reset_index(drop=True)

col_to_drop = train.columns[train.columns.str.endswith('_cat')]

col_to_dummify = train.columns[train.columns.str.endswith('_cat')].astype(str).tolist()



for col in col_to_dummify:

    dummy = pd.get_dummies(train_test[col].astype('category'))

    columns = dummy.columns.astype(str).tolist()

    columns = [col + '_' + w for w in columns]

    dummy.columns = columns

    train_test = pd.concat((train_test, dummy), axis=1)



train_test.drop(col_to_dummify, axis=1, inplace=True)

train_test_scaled, scaler = scale_data(train_test)

train = train_test_scaled[:n_train, :]

test = train_test_scaled[n_train:, :]

print('\n Shape of processed train data:', train.shape)

print(' Shape of processed test data:', test.shape)



# We really didn't need to load the test data in the first place unless you are planning to make

# a prediction at the end of this run.

# del test

# gc.collect()
# dtrain = xgb.DMatrix(train, label = target)



sss = StratifiedShuffleSplit(target, random_state=1001, test_size=0.75)

for train_index, test_index in sss:

    break

X_train, y_train = train[train_index], target[train_index]

del train, target

gc.collect()

dtrain = xgb.DMatrix(X_train, label = y_train)

XGB_BO = BayesianOptimization(XGB_CV, {

                                     'max_depth': (2, 12),

                                     'gamma': (0.001, 10.0),

                                     'min_child_weight': (0, 20),

                                     'max_delta_step': (0, 10),

                                     'subsample': (0.4, 1.0),

                                     'colsample_bytree' :(0.4, 1.0)

                                    })
XGB_BO.explore({

              'max_depth':            [3, 8, 3, 8, 8, 3, 8, 3],

              'gamma':                [0.5, 8, 0.2, 9, 0.5, 8, 0.2, 9],

              'min_child_weight':     [0.2, 0.2, 0.2, 0.2, 12, 12, 12, 12],

              'max_delta_step':       [1, 2, 2, 1, 2, 1, 1, 2],

              'subsample':            [0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],

              'colsample_bytree':     [0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],

              })
print('-'*130)

print('-'*130, file=log_file)

log_file.flush()



with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    XGB_BO.maximize(init_points=2, n_iter=5, acq='ei', xi=0.0)



# XGB_BO.maximize(init_points=10, n_iter=50, acq='ei', xi=0.0)

# XGB_BO.maximize(init_points=10, n_iter=50, acq='ei', xi=0.01)

# XGB_BO.maximize(init_points=10, n_iter=50, acq='ucb', kappa=10)

# XGB_BO.maximize(init_points=10, n_iter=50, acq='ucb', kappa=1)
print('-'*130)

print('Final Results')

print('Maximum XGBOOST value: %f' % XGB_BO.res['max']['max_val'])

print('Best XGBOOST parameters: ', XGB_BO.res['max']['max_params'])

print('-'*130, file=log_file)

print('Final Result:', file=log_file)

print('Maximum XGBOOST value: %f' % XGB_BO.res['max']['max_val'], file=log_file)

print('Best XGBOOST parameters: ', XGB_BO.res['max']['max_params'], file=log_file)

log_file.flush()

log_file.close()



history_df = pd.DataFrame(XGB_BO.res['all']['params'])

history_df2 = pd.DataFrame(XGB_BO.res['all']['values'])

history_df = pd.concat((history_df, history_df2), axis=1)

history_df.rename(columns = { 0 : 'gini'}, inplace=True)

history_df['AUC'] = ( history_df['gini'] + 1 ) / 2

history_df.to_csv('Porto-AUC-5fold-XGB-run-01-v1-grid.csv')