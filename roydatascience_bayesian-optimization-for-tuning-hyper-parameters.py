from bayes_opt import BayesianOptimization
import numpy as np 

import pandas as pd 

# Data processing, metrics and modeling

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, StratifiedKFold,KFold

from bayes_opt import BayesianOptimization

from datetime import datetime

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve

from sklearn import metrics

from sklearn import preprocessing

# Lgbm

import lightgbm as lgb

# Suppr warning

import warnings

warnings.filterwarnings("ignore")



import itertools

from scipy import interp



# Plots

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import rcParams

import os

print(os.listdir('../input/'))

Random_Seed = 4520
%%time

DataFile = pd.read_csv('../input/creditcard.csv')

train_df = DataFile.drop(['Time'], axis=1)
features = list(train_df)

features.remove('Class')
#cut tr and val

bayesian_tr_idx, bayesian_val_idx = train_test_split(train_df, test_size = 0.3, random_state = 42, stratify = train_df['Class'])

bayesian_tr_idx = bayesian_tr_idx.index

bayesian_val_idx = bayesian_val_idx.index
# Take the hyper parameters you want to consider



paramsLGB = {

    'learning_rate': (0.001,0.005),

    'num_leaves': (50, 500), 

    'bagging_fraction' : (0.1, 0.9),

    'feature_fraction' : (0.1, 0.9),

    'min_child_weight': (0.00001, 0.01),   

    'min_data_in_leaf': (20, 70),

    'max_depth':(-1,50),

    'reg_alpha': (1, 2), 

    'reg_lambda': (1, 2)

    

}
def LGB_bayesian(

    learning_rate,

    num_leaves, 

    bagging_fraction,

    feature_fraction,

    min_child_weight, 

    min_data_in_leaf,

    max_depth,

    reg_alpha,

    reg_lambda

     ):

    

    # LightGBM expects next three parameters need to be integer. 

    num_leaves = int(num_leaves)

    min_data_in_leaf = int(min_data_in_leaf)

    max_depth = int(max_depth)



    assert type(num_leaves) == int

    assert type(min_data_in_leaf) == int

    assert type(max_depth) == int

    



    param = {

              'num_leaves': num_leaves, 

              'min_data_in_leaf': min_data_in_leaf,

              'min_child_weight': min_child_weight,

              'bagging_fraction' : bagging_fraction,

              'feature_fraction' : feature_fraction,

              'learning_rate' : learning_rate,

              'max_depth': max_depth,

              'reg_alpha': reg_alpha,

              'reg_lambda': reg_lambda,

              'objective': 'binary',

              'save_binary': True,

              'seed': Random_Seed,

              'feature_fraction_seed': Random_Seed,

              'bagging_seed': Random_Seed,

              'drop_seed': Random_Seed,

              'data_random_seed': Random_Seed,

              'boosting_type': 'gbdt',

              'verbose': 1,

              'is_unbalance': False,

              'boost_from_average': True,

              'metric':'auc'}    

    

    oof = np.zeros(len(train_df))

    trn_data= lgb.Dataset(train_df.iloc[bayesian_tr_idx][features].values, label=train_df.iloc[bayesian_tr_idx]['Class'].values)

    val_data= lgb.Dataset(train_df.iloc[bayesian_val_idx][features].values, label=train_df.iloc[bayesian_val_idx]['Class'].values)



    clf = lgb.train(param, trn_data,  num_boost_round=50, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds = 50)

    

    oof[bayesian_val_idx]  = clf.predict(train_df.iloc[bayesian_val_idx][features].values, num_iteration=clf.best_iteration)  

    

    score = roc_auc_score(train_df.iloc[bayesian_val_idx]['Class'].values, oof[bayesian_val_idx])



    return score
LGB_BO = BayesianOptimization(LGB_bayesian, paramsLGB, random_state=42)
print(LGB_BO.space.keys)
init_points = 9

n_iter = 15
print('-' * 130)

with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
LGB_BO.max['target']
LGB_BO.max['params']
param_lgb = {

        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), 

        'num_leaves': int(LGB_BO.max['params']['num_leaves']), 

        'learning_rate': LGB_BO.max['params']['learning_rate'],

        'min_child_weight': LGB_BO.max['params']['min_child_weight'],

        'bagging_fraction': LGB_BO.max['params']['bagging_fraction'], 

        'feature_fraction': LGB_BO.max['params']['feature_fraction'],

        'reg_lambda': LGB_BO.max['params']['reg_lambda'],

        'reg_alpha': LGB_BO.max['params']['reg_alpha'],

        'max_depth': int(LGB_BO.max['params']['max_depth']), 

        'objective': 'binary',

        'save_binary': True,

        'seed': Random_Seed,

        'feature_fraction_seed': Random_Seed,

        'bagging_seed': Random_Seed,

        'drop_seed': Random_Seed,

        'data_random_seed': Random_Seed,

        'boosting_type': 'gbdt',

        'verbose': 1,

        'is_unbalance': False,

        'boost_from_average': True,

        'metric':'auc'

    }