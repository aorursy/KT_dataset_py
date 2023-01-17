!pip install bayesian-optimization
# importing libraries

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import roc_auc_score

from scipy.stats import rankdata

import gc

import warnings

import os



pd.set_option('display.max_columns', 200)
!ls "../input/santander-customer-transaction-prediction/"
train_df = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")

test_df = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")
train_df.shape, test_df.shape
train_df.head()
test_df.head()
predictors = train_df.columns.tolist()[2:]
train_df.target.value_counts(normalize = True)
train_target = train_df['target']

train = train_df.loc[:,predictors]
# 50% rows used for stratified split in order to find optimal parameters

train_index, valid_index = list(StratifiedKFold(n_splits = 2, shuffle = True, random_state = 1).split(train,train_target.values))[0]
## Blackbox function to optimize using lightgbm



def LGB_bayesian(

                num_leaves,

                min_data_in_leaf,

                learning_rate,

                min_sum_hessian_in_leaf,

                feature_fraction,

                lambda_l1,

                lambda_l2,

                min_gain_to_split,

                max_depth):

    

    #Below params should be always int

    num_leaves = int(num_leaves)

    min_data_in_leaf = int(min_data_in_leaf)

    max_depth = int(max_depth)

    

    assert type(num_leaves) == int

    assert type(min_data_in_leaf) == int

    assert type(max_depth) == int

    

    param = {

        'num_leaves' : num_leaves,

        'min_data_in_leaf':min_data_in_leaf,

        'max_bin' : 63,#max number of bins that feature values will be bucketed in

        'learning_rate':learning_rate,

        'min_sum_hessian_in_leaf' :min_sum_hessian_in_leaf,

        'bagging_fraction' : 1.0 ,#this will randomly select part of data without resampling

        'bagging_freq':5, #Note: to enable bagging, bagging_freq should be set to a non zero value as well

        'feature_fraction':feature_fraction,

        'lambda_l1':lambda_l1,

        'lambda_l2' : lambda_l2,

        'min_gain_to_split':min_gain_to_split, #the minimal gain to perform split

        'max_depth':max_depth,

        'save_binary':True, #if true, LightGBM will save the dataset (including validation data) to a binary file. This speed ups the data loading for the next time

        'seed':1337,

        'feature_fraction_seed':1337,

        'bagging_seed' : 1337,

        'drop_seed':1337,

        'data_random_seed':1337,

        'objective':'binary',

        'boosting_type' : 'gbdt',

        'verbose' : 1,

        'metric' : 'auc',

        'is_unbalance' : True,

        'boost_from_average':True  

    }

    

    xg_train = lgb.Dataset(train.iloc[train_index][predictors].values, 

                           label = train_target[train_index].values, 

                           feature_name = predictors, 

                           free_raw_data = False #If True, raw data is freed after constructing inner Dataset.

                          )

    

    xg_valid = lgb.Dataset(train.iloc[valid_index][predictors].values,

                          label = train_target[valid_index].values,

                          feature_name= predictors,

                          free_raw_data = False # If True, raw data is freed after constructing inner Dataset.

                          )

    

    num_round = 5000

    clf = lgb.train(param, xg_train, num_round, valid_sets = [xg_valid], verbose_eval = 200,

                   early_stopping_rounds=50)

    

    predictions = clf.predict(train.iloc[valid_index][predictors].values, num_iterations = clf.best_iteration)

    

    score = roc_auc_score(train_target[valid_index].values, predictions)

    return score
##Bounds for params



bounds_LGB = {

    'num_leaves' : (5,20),

    'min_data_in_leaf': (5,20),

    'learning_rate' : (0.01, 0.3),

    'min_sum_hessian_in_leaf' : (0.00001,0.01),

    'feature_fraction': (0.05,0.5),

    'lambda_l1' : (0,5.0),

    'lambda_l2': (0,5.0),

    'min_gain_to_split': (0,1.0),

    'max_depth':(3,15)

}
from bayes_opt import BayesianOptimization
LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state = 13)
print(LGB_BO.space.keys)
init_points = 5 # Number of inital random runs for exploration

n_iter = 5 #Number of bayesian optimization to perform after initial runs

print('-'*130)

with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    LGB_BO.maximize(init_points = init_points, n_iter = n_iter, acq ='ucb',xi =0.0, alpha = 1e-6)

    
LGB_BO.max['target']
LGB_BO.max['params']
#You can probe the LGB_bayesian function, if you have an idea of the optimal parameters 

LGB_BO.probe(

    params={'feature_fraction': 0.1403, 

            'lambda_l1': 4.218, 

            'lambda_l2': 1.734, 

            'learning_rate': 0.07, 

            'max_depth': 14, 

            'min_data_in_leaf': 17, 

            'min_gain_to_split': 0.1501, 

            'min_sum_hessian_in_leaf': 0.000446, 

            'num_leaves': 6},

    lazy=True, # 

)
LGB_BO.maximize(init_points = 0, n_iter = 0)
for i, res in enumerate(LGB_BO.res):

    print("Iteration : {} \n\t {}".format(i, res))
LGB_BO.max['target']
LGB_BO.max['params']
# Training lightGBM model

param_lgb = {

        'num_leaves': int(LGB_BO.max['params']['num_leaves']), # remember to int here

        'max_bin': 63,

        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), # remember to int here

        'learning_rate': LGB_BO.max['params']['learning_rate'],

        'min_sum_hessian_in_leaf': LGB_BO.max['params']['min_sum_hessian_in_leaf'],

        'bagging_fraction': 1.0, 

        'bagging_freq': 5, 

        'feature_fraction': LGB_BO.max['params']['feature_fraction'],

        'lambda_l1': LGB_BO.max['params']['lambda_l1'],

        'lambda_l2': LGB_BO.max['params']['lambda_l2'],

        'min_gain_to_split': LGB_BO.max['params']['min_gain_to_split'],

        'max_depth': int(LGB_BO.max['params']['max_depth']), # remember to int here

        'save_binary': True,

        'seed': 1337,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

        'boost_from_average': False,

    }
nfolds = 5
gc.collect()
skf = StratifiedKFold(n_splits = nfolds, shuffle = True, random_state = 2019)
oof = np.zeros(len(train_df))

predictions = np.zeros((len(test_df),nfolds))
i = 1

for train_index, valid_index in skf.split(train, train_target.values):

    print("\nfold {}".format(i))

    xg_train = lgb.Dataset(train.iloc[train_index][predictors].values,

                           label=train_target.iloc[train_index].values,

                           feature_name=predictors,

                           free_raw_data = False

                           )

    xg_valid = lgb.Dataset(train.iloc[valid_index][predictors].values,

                           label=train_target.iloc[valid_index].values,

                           feature_name=predictors,

                           free_raw_data = False

                           )   



    

    clf = lgb.train(param_lgb, xg_train, 5000, valid_sets = [xg_valid], verbose_eval=250, early_stopping_rounds = 50)

    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration) 

    

    predictions[:,i-1] += clf.predict(test_df[predictors], num_iteration=clf.best_iteration)

    i = i + 1



print("\n\nCV AUC: {:<0.2f}".format(roc_auc_score(train_df.target.values, oof)))
print("\n\nCV AUC: {:<0.2f}".format(roc_auc_score(train_df.target.values, oof)))
predictions
print("Rank averaging on", nfolds, "fold predictions")

rank_predictions = np.zeros((predictions.shape[0],1))

for i in range(nfolds):

    rank_predictions[:, 0] = np.add(rank_predictions[:, 0], rankdata(predictions[:, i].reshape(-1,1))/rank_predictions.shape[0]) 



rank_predictions /= nfolds
sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub_df["target"] = rank_predictions

sub_df[:10]
sub_df.to_csv('submission.csv', index=False)