import xgboost as xgb

import pandas as pd

import numpy as np

from uuid import uuid4

from sys import exit
from hyperopt.pyll.base import scope

from hyperopt.pyll.stochastic import sample

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.model_selection import GridSearchCV
# Setting working directory



path = '../input/'
#load files

train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')
test_id = pd.read_csv(path + 'test.csv')['id']
X = train.drop("target", axis=1)

y = train.target
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=0.3)
num_train, num_feature = X_train.shape
# create dataset for lightgbm

# if you want to re-use data, remember to set free_raw_data=False

xgb_train = xgb.DMatrix(X_train.values, y_train.values)

xgb_eval  = xgb.DMatrix(X_val.values, y_val.values)
xgb_test   = xgb.DMatrix(test.values)
import random

import itertools

N_HYPEROPT_PROBES = 100 #change to 5000

EARLY_STOPPING = 50 #change to 80

HOLDOUT_SEED = 123456

HOLDOUT_SIZE = 0.10

HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest

SEED0 = random.randint(1,1000000000)

NB_CV_FOLDS = 3 #chagne to 5
obj_call_count = 0

cur_best_score = 0
def objective(space):

    

    global obj_call_count, cur_best_score, X_train, y_train, test, X_val, y_val



    

    obj_call_count += 1

    print('\nXGBoost objective call #{} cur_best_score={:7.5f}'.format(obj_call_count,cur_best_score) )



    sorted_params = sorted(space.items(), key=lambda z: z[0])

    print('Params:', str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params if not k.startswith('column:')]))





    xgb_params = sample(space)

       

    

    mdl = xgb.cv(

                        xgb_params,

                        xgb_train,

                        num_boost_round = 10,##change to 750,

                        nfold=NB_CV_FOLDS,

                        #metrics='binary_logloss',

                        stratified=False,

                        #fobj=None,

                        #feval=None,

                        #init_model=None,

                        #feature_name='auto',

                        early_stopping_rounds=EARLY_STOPPING,

                        #fpreproc=None,

                        verbose_eval=100,

                        show_stdv=False,

                        )



           

    

    n_rounds = len(mdl["test-auc-mean"])

    cv_score = mdl["test-auc-mean"][n_rounds-1]



    print( 'CV finished n_rounds={} cv_score={:7.5f}'.format( n_rounds, cv_score ) )

    

    gbm_model = xgb.train(

                        xgb_params,

                        xgb_train,

                        num_boost_round=n_rounds,

                        # metrics='mlogloss',

                        # valid_names=None,

                        # fobj=None,

                        # init_model=None,

                        # feature_name='auto',

                        # categorical_feature='auto',

                        # early_stopping_rounds=None,

                        # evals_result=None,

                        verbose_eval=False

                        # learning_rates=None,

                        # keep_training_booster=False,

                        # callbacks=None)

                         )

    

    predictions = gbm_model.predict(xgb_eval,

                                    ntree_limit =n_rounds)

    

    score = roc_auc_score(y_val, predictions)

    print('valid score={}'.format(score))

    

    

#     do_submit = score > 0.63



    if score > cur_best_score:

        cur_best_score = score

        print('NEW BEST SCORE={}'.format(cur_best_score))

#         do_submit = True



#     if do_submit:

#         submit_guid = uuid4()



#         print('Compute submissions guid={}'.format(submit_guid))



#         y_submission = gbm_model.predict(xgb_test, ntree_limit = n_rounds)

#         submission_filename = 'xgboost_score={:13.11f}_submission_guid={}.csv'.format(score,submit_guid)

#         pd.DataFrame(

#         {'id':test_id, 'target':y_submission}

#         ).to_csv(submission_filename, index=False)

       

    loss = 1 - score

    return {'loss': loss, 'status': STATUS_OK}



   
space ={

    'booster '    : 'gbtree',       

    'objective'   : 'binary:logistic',

    'eval_metric' : 'auc',

     

    'max_depth'   : hp.choice("max_depth",        np.arange(4, 7,    dtype=int)),  

   

    'alpha'       : hp.uniform('alpha', 1e-4, 1e-6 ),

    'lambda'      : hp.uniform('lambda', 1e-4, 1e-6 ),

    

    'min_child_weight ': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),

    'learning_rate'    : hp.loguniform('learning_rate', -6.9, -2.3),

    

    'seed'             : hp.randint('seed',2000000)

   }
trials = Trials()

best = fmin(fn=objective,

                     space=space,

                     algo=HYPEROPT_ALGO,

                     max_evals=N_HYPEROPT_PROBES,

                     trials=trials,

                     verbose=1)



print('-'*50)

print('The best params:')

print( best )

print('\n\n')