import numpy as np

import pandas as pd

from time import gmtime, strftime

import gc



from sklearn.model_selection import (train_test_split, GridSearchCV)

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from tqdm import tqdm

from sklearn.metrics import (roc_curve, auc, accuracy_score)
# read the first 99 rows for demo

train = pd.read_csv('../input/train.csv', nrows=99)

test = pd.read_csv('../input/test.csv',nrows=99)

songs = pd.read_csv('../input/songs.csv')

members = pd.read_csv('../input/members.csv')



# Merge datasets with song attributes

song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']

train = train.merge(songs[song_cols], on='song_id', how='left')

test = test.merge(songs[song_cols], on='song_id', how='left')



# Merge datasets with member features

members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))

members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))

members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))



members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))

members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))

members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))

members = members.drop(['registration_init_time'], axis=1)



members_cols = members.columns

train = train.merge(members[members_cols], on='msno', how='left')

test = test.merge(members[members_cols], on='msno', how='left')



train = train.fillna(-1)

test = test.fillna(-1)



del members, songs; gc.collect();



cols = list(train.columns)

cols.remove('target')



for col in tqdm(cols):

    if train[col].dtype == 'object':

        train[col] = train[col].apply(str)

        test[col] = test[col].apply(str)



        le = LabelEncoder()

        train_vals = list(train[col].unique())

        test_vals = list(test[col].unique())

        le.fit(train_vals + test_vals)

        train[col] = le.transform(train[col])

        test[col] = le.transform(test[col])



X = np.array(train.drop(['target'], axis=1))

y = train['target'].values
params = {

    'application': 'binary', # for binary classification

#     'num_class' : 1, # used for multi-classes

    'boosting': 'gbdt', # traditional gradient boosting decision tree

    'num_iterations': 100, 

    'learning_rate': 0.05,

    'num_leaves': 62,

    'device': 'cpu', # you can use GPU to achieve faster learning

    'max_depth': -1, # <0 means no limit

    'max_bin': 510, # Small number of bins may reduce training accuracy but can deal with over-fitting

    'lambda_l1': 5, # L1 regularization

    'lambda_l2': 10, # L2 regularization

    'metric' : 'binary_error',

    'subsample_for_bin': 200, # number of samples for constructing bins

    'subsample': 1, # subsample ratio of the training instance

    'colsample_bytree': 0.8, # subsample ratio of columns when constructing the tree

    'min_split_gain': 0.5, # minimum loss reduction required to make further partition on a leaf node of the tree

    'min_child_weight': 1, # minimum sum of instance weight (hessian) needed in a leaf

    'min_child_samples': 5# minimum number of data needed in a leaf

}



# Initiate classifier to use

mdl = lgb.LGBMClassifier(boosting_type= 'gbdt', 

          objective = 'binary', 

          n_jobs = 5, 

          silent = True,

          max_depth = params['max_depth'],

          max_bin = params['max_bin'], 

          subsample_for_bin = params['subsample_for_bin'],

          subsample = params['subsample'], 

          min_split_gain = params['min_split_gain'], 

          min_child_weight = params['min_child_weight'], 

          min_child_samples = params['min_child_samples'])



# To view the default model parameters:

mdl.get_params().keys()



gridParams = {

    'learning_rate': [0.005, 0.01],

    'n_estimators': [8,16,24],

    'num_leaves': [6,8,12,16], # large num_leaves helps improve accuracy but might lead to over-fitting

    'boosting_type' : ['gbdt', 'dart'], # for better accuracy -> try dart

    'objective' : ['binary'],

    'max_bin':[255, 510], # large max_bin helps improve accuracy but might slow down training progress

    'random_state' : [500],

    'colsample_bytree' : [0.64, 0.65, 0.66],

    'subsample' : [0.7,0.75],

    'reg_alpha' : [1,1.2],

    'reg_lambda' : [1,1.2,1.4],

    }



grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)

# Run the grid

grid.fit(X, y)



# Print the best parameters found

print(grid.best_params_)

print(grid.best_score_)



params['colsample_bytree'] = grid.best_params_['colsample_bytree']

params['learning_rate'] = grid.best_params_['learning_rate'] 

params['max_bin'] = grid.best_params_['max_bin']

params['num_leaves'] = grid.best_params_['num_leaves']

params['reg_alpha'] = grid.best_params_['reg_alpha']

params['reg_lambda'] = grid.best_params_['reg_lambda']

params['subsample'] = grid.best_params_['subsample']





X_test = np.array(test.drop(['id'], axis=1))

ids = test['id'].values





X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state = 12)

    

del X, y; gc.collect();



d_train = lgb.Dataset(X_train, label=y_train)

d_valid = lgb.Dataset(X_valid, label=y_valid) 



watchlist = [d_train, d_valid]





model = lgb.train(params, train_set=d_train, num_boost_round=1000, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=4)



p_test = model.predict(X_test)



subm = pd.DataFrame()

subm['id'] = ids

subm['target'] = p_test

submName = strftime("%Y%m%d%H%M%S", gmtime()) + '_submission.csv.gz'

subm.to_csv(submName, compression = 'gzip', index=False, float_format = '%.5f')