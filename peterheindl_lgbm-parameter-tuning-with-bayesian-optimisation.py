import lightgbm as lgb

import pandas as pd

from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.metrics import roc_auc_score

from sklearn import preprocessing

import random, json

from bayes_opt import BayesianOptimization



# Load data for training 

mydata = pd.read_csv('../input/train.csv', sep=',')

mydata = mydata.drop('ID_code', 1)

# Load prediction data

preddata = pd.read_csv('../input/test.csv', sep=',')

predids = preddata[['ID_code']] 

iddf = preddata[['ID_code']] 

preddata = preddata.drop('ID_code', 1)



# Test train split

df_train, df_test = train_test_split(mydata, test_size=0.3, random_state=76)

# Same random state to make sure rows merge

y_train = df_train['target']

y_test = df_test['target']

X_train = df_train.drop('target', 1)

X_test = df_test.drop('target', 1)



# Scale data

scaler = preprocessing.StandardScaler()

scaled_df = scaler.fit_transform(X_train)

X_train = pd.DataFrame(scaled_df)

scaled_df = scaler.fit_transform(X_test)

X_test = pd.DataFrame(scaled_df)

scaled_df = scaler.fit_transform(preddata)

preddata = pd.DataFrame(scaled_df)



# Create dataset for lightgbm input

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
def lgbmfunc(min_split_gain, min_child_weight):

    params = {

        'boost_from_average' : False,

        'objective' :'binary',

        'learning_rate' : 0.002,

        'num_leaves' : 24, 

        'feature_fraction': 0.07, 

        'bagging_fraction': 0.2, 

        'bagging_freq': 3, 

        'max_bin' : 255, #default 255

        'scale_pos_weight' : 1,  #default = 1 (used only in binary application) weight of labels with positive class

        'boosting_type' : 'gbdt',

        'metric': 'auc',

        'num_threads' : 4,

        'tree_learner': 'serial', 

        'boost_from_average':'false',

        'min_child_samples': 3

    }

    params['min_split_gain'] = min_split_gain

    params['min_child_weight'] = min_child_weight



    # LGBM CV

    cv_results = lgb.cv(params, lgb_train, num_boost_round=500, nfold=5, early_stopping_rounds=50, metrics='auc')

    return max(cv_results['auc-mean'])



pbounds = { 

        'min_split_gain': (0.2, 0.3), 

        'min_child_weight': (0.1, 0.3),

    }



optimizer = BayesianOptimization(

    f=lgbmfunc,

    pbounds=pbounds,

    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent

    random_state=1,

)



optimizer.maximize(

    init_points=2,

    n_iter=10,

)

print("list of results")

for i, res in enumerate(optimizer.res):

    print("Iteration {}: \n\t{}".format(i, res))



print("best result")

print(optimizer.max)