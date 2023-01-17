import os

import random

import math



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import sklearn

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.metrics import auc, roc_auc_score



from lightgbm import LGBMClassifier

import lightgbm as lgb



from hyperopt import hp, tpe, Trials, fmin

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

submission = pd.read_csv("../input/sample_submission.csv")
train.shape, test.shape
train.head()
np.sum(train['target']) / len(train['target'])
X_train, X_test, y_train, y_test = train_test_split(

                                                        train.drop(['ID_code', 'target'], axis = 1),

                                                        train['target'],

                                                        test_size = 0.15,

                                                        random_state = 42,

                                                        stratify = train['target']

                                                    )
np.sum(y_train) / len(y_train), np.sum(y_test) / len(y_test)
def objective(params):

    

    n_folds = 3

    params = {

                'n_estimators': int(params['n_estimators']),

                'num_leaves': int(params['num_leaves']),

                'learning_rate': float(params['learning_rate']),

                'subsample_for_bin': int(params['subsample_for_bin']),

                'min_child_samples': int(params['min_child_samples']),

                'reg_alpha': float(params['reg_alpha']),

                'reg_lambda': float(params['reg_lambda'])

             }

    

    clf = LGBMClassifier(**params)

    score1 = cross_val_score(clf, X_train, y_train, scoring = 'roc_auc', cv = StratifiedKFold(n_splits = n_folds)).mean()

    clf.fit(

                X_train, y_train,

                eval_set = [(X_test, y_test)],

                eval_metric = 'auc',

                early_stopping_rounds = 3,

                verbose = False,

            )

    score2 = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    score = 2 / ((1 / score1) + (1 / score2))



    return 1 - score
space = {

            'n_estimators': hp.quniform('n_estimators', 50, 1500, 25),

            'num_leaves': hp.quniform('num_leaves', 30, 150, 1),

            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),

            'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),

            'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),

            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),

            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)

        }

tpe_algo = tpe.suggest

tpe_trials = Trials()



# tpe_best = fmin(fn = objective, space = space, algo = tpe_algo, trials = tpe_trials, 

#                 max_evals = 15, rstate = np.random.RandomState(42))
# tpe_best
# params = {

#             'n_estimators': int(tpe_best['n_estimators']),

#             'num_leaves': int(tpe_best['num_leaves']),

#             'learning_rate': float(tpe_best['learning_rate']),

#             'subsample_for_bin': int(tpe_best['subsample_for_bin']),

#             'min_child_samples': int(tpe_best['min_child_samples']),

#             'reg_alpha': float(tpe_best['reg_alpha']),

#             'reg_lambda': float(tpe_best['reg_lambda'])

#          }
params = {

            'bagging_freq': 5,

            'bagging_fraction': 0.331,

            'boost_from_average':'false',

            'boost': 'gbdt',

            'feature_fraction': 0.0405,

            'learning_rate': 0.0083,

            'max_depth': -1,

            'metric':'auc',

            'min_data_in_leaf': 80,

            'min_sum_hessian_in_leaf': 10.0,

            'num_leaves': 13,

            'num_threads': 8,

            'tree_learner': 'serial',

            'objective': 'binary',

            'verbosity': 1

        }



params = {

             'num_leaves': 8,

             'min_data_in_leaf': 42,

             'objective': 'binary',

             'max_depth': 16,

             'learning_rate': 0.0123,

             'boosting': 'gbdt',

             'bagging_freq': 5,

             'feature_fraction': 0.8201,

             'bagging_seed': 11,

             'reg_alpha': 1.728910519108444,

             'reg_lambda': 4.9847051755586085,

             'random_state': 42,

             'metric': 'auc',

             'verbosity': -1,

             'subsample': 0.81,

             'min_gain_to_split': 0.01077313523861969,

             'min_child_weight': 19.428902804238373,

             'num_threads': 4

         }



lgb_base = LGBMClassifier(**params, n_estimators = 20000)



lgb_base.fit(

                X_train, y_train,

                eval_set = [(X_test, y_test)],

                eval_metric = 'auc',

                early_stopping_rounds = 200,

                verbose = 1000,

            )
print("Training AUC:")

print(roc_auc_score(y_train, lgb_base.predict_proba(X_train)[:, 1]))

print("\n")

print("Testing AUC:")

print(roc_auc_score(y_test, lgb_base.predict_proba(X_test)[:, 1]))
submission['target'] = 1 - lgb_base.predict_proba(test.drop('ID_code', axis = 1))
submission.head()
submission.to_csv("submission_hyperparams_borrowed.csv", index = False)