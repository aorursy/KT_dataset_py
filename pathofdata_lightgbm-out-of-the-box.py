import os

from time import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import ParameterGrid, train_test_split, StratifiedKFold, cross_val_score

from sklearn.metrics import make_scorer

from skopt import BayesSearchCV

from skopt import gp_minimize

from skopt.space import Real, Categorical, Integer

from skopt.utils import use_named_args

from skopt.plots import plot_convergence, plot_objective, plot_evaluations

from time import time
DF = '../input/train.csv'

TEST_DF = '../input/test.csv'

SUB_DF = '../input/sample_submission.csv'
data = pd.read_csv(DF)

data.head()
X = data.drop(['ID_code', 'target'], axis=1).values

scaler = MinMaxScaler(copy=False)

scaler.fit_transform(X)

Y = data.target.values

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
import keras as k

from keras.models import Sequential

from keras.layers import *



def linear_clf(x_train_, y_train_, x_val_, y_val_):

    model = Sequential()

    model.add(Dense(1, activation='sigmoid', input_dim=200))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    history = model.fit(x_train_, y_train_, epochs=10, validation_data=(x_val_, y_val_))

    return model



skf = StratifiedKFold(n_splits=5, shuffle=True)



for i, (train_index, test_index) in enumerate(skf.split(X,Y)):

    

    # Create data for this fold

    y_train, y_valid = Y[train_index], Y[test_index]

    X_train, X_valid = X[train_index,:], X[test_index,:]

        

    print( "\nFold ", i)



    linear_m = linear_clf(X_train, y_train,

                         X_valid, y_valid)

test_df = pd.read_csv(TEST_DF)

test_df.head()

X_test = test_df.drop(['ID_code'], axis=1)

X_test = scaler.fit_transform(X_test.values)

predictions = linear_m.predict(X_test)



submission_df = pd.read_csv(SUB_DF)

submission_df.target = predictions

submission_df.to_csv('submission.csv', index=False)

submission_df.head()
# params = {

#     'num_leaves': 6,

#     'max_bin': 263,

#     'min_data_in_leaf': 90,

#     'learning_rate': 0.01,

#     'min_sum_hessian_in_leaf': 0.000446,

#     'bagging_fraction': 0.55,

#     'bagging_freq': 5,

#     'max_depth': 14,

#     'save_binary': True,

#     'seed': 31452,

#     'drop_seed': 31452,

#     'data_random_seed': 31452,

#     'objective': 'binary',

#     'boosting_type': 'gbdt',

#     'verbose': 1,

#     'metric': 'auc',

#     'is_unbalance': True,

#     'boost_from_average': False,

# }
# skf = StratifiedKFold(n_splits=5, shuffle=True)



# for i, (train_index, test_index) in enumerate(skf.split(X,Y)):

    

#     # Create data for this fold

#     y_train, y_valid = Y[train_index], Y[test_index]

#     X_train, X_valid = X[train_index,:], X[test_index,:]

        

#     print( "\nFold ", i)



#     # Running models for this fold

    

#     # ->LightGBM

#     lgb_gbm = lgb.train(params, 

#                           lgb.Dataset(X_train, label=y_train), 

#                           MAX_ROUNDS, 

#                           lgb.Dataset(X_valid, label=y_valid), 

#                           verbose_eval=False, 

#                           #feval= auc, 

#                           early_stopping_rounds=350)

    

#     print( " Best iteration lgb = ", lgb_gbm.best_iteration)

    

#     # Storing and reporting results of the fold

#     lgb_iter1 = np.append(lgb_iter1, lgb_gbm.best_iteration)

#     pred = lgb_gbm.predict(X_valid, num_iteration=lgb_gbm.best_iteration)

#     auc = roc_auc_score(y_valid, pred)

#     print('lgb score: ', auc)

#     lgb_ap1 = np.append(lgb_ap1, auc)

    
# auc = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)

# skf = StratifiedKFold(n_splits=2, shuffle=True)
# counter = 0

# def onstep(res):

#     global counter

#     x0 = res.x_iters   # List of input points

#     y0 = res.func_vals # Evaluation of input points

# #     print('Last eval: ', x0[-1], 

# #           ' - Score ', y0[-1])

#     print('Current iter: ', counter, 

#           ' - Score ', res.fun, 

#           ' - Args: ', res.x)

#     counter += 1



# dimensions = [Real(0.01, 1.0, name="learning_rate"),

#               Integer(2, 20, name="num_leaves"),

#               Integer(30, 300, name="Max_bin"),

#               Real(0.000001, 1.0, name="min_sum_hessian_in_leaf"),

#               Integer(10, 1500, name="n_estimators"),

#               Real(0.5, 1.0, name="subsample"),

#               Integer(2, 100, name="min_data_in_leaf")]
# @use_named_args(dimensions=dimensions) 

# def objective(**params):

#     model = lgb.LGBMClassifier(boosting_type='gbdt',

#                                objective='binary',

#         #                        num_rounds = 15000,

# #                                num_leaves= 6,

# #                                max_bin= 63,

# #                                min_data_in_leaf= 90,

# #                                learning_rate= 0.01,

# #                                min_sum_hessian_in_leaf= 0.000446,

# #                                bagging_fraction= 0.55,

#                                max_depth= 14,

#                                save_binary= True,

#                                seed= 31452,

#                                drop_seed= 31452,

#                                data_random_seed= 31452,

#                                metric= 'auc',

#                                is_unbalance= True,

#                                boost_from_average= False,

#                                n_jobs=-1,

#                                verbose=0)

        

#     model.set_params(**params)

#     return -np.mean(cross_val_score(model, 

#                                     X, Y, 

#                                     cv=skf, 

#                                     n_jobs=-1,

#                                     scoring=auc))
# gp_round = gp_minimize(func=objective,

#                        dimensions=dimensions,

#                        acq_func='gp_hedge',

#                        n_calls=100,

#                        callback=[onstep])
# plot_convergence(gp_round)
# Stacker class from https://www.kaggle.com/yekenot/simple-stacker-lb-0-284



# class Ensemble(object):

#     def __init__(self, n_splits, stacker, base_models):

#         self.n_splits = n_splits

#         self.stacker = stacker

#         self.base_models = base_models



#     def fit_predict(self, X, y, T):

#         X = np.array(X)

#         y = np.array(y)

#         T = np.array(T)



#         folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True).split(X, y))



#         S_train = np.zeros((X.shape[0], len(self.base_models)))

#         S_test = np.zeros((T.shape[0], len(self.base_models)))

#         for i, clf in enumerate(self.base_models):



#             S_test_i = np.zeros((T.shape[0], self.n_splits))



#             for j, (train_idx, test_idx) in enumerate(folds):

#                 X_train = X[train_idx]

#                 y_train = y[train_idx]

#                 X_holdout = X[test_idx]

#                 y_holdout = y[test_idx]



#                 print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))

#                 clf.fit(X_train, y_train)

# #                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')

# #                print("    cross_score: %.5f" % (cross_score.mean()))

#                 y_pred = clf.predict_proba(X_holdout)[:,1]                



#                 S_train[test_idx, i] = y_pred

#                 S_test_i[:, j] = clf.predict_proba(T)[:,1]

#             S_test[:, i] = S_test_i.mean(axis=1)



#         results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')

#         print("Stacker score: %.5f" % (results.mean()))



#         self.stacker.fit(S_train, y)

#         res = self.stacker.predict_proba(S_test)[:,1]

#         return res
# lgb_params1 = {}

# lgb_params1['boosting_type']= 'gbdt'

# lgb_params1['objective'] = 'binary'

# lgb_params1['bagging_fraction']= 0.55

# lgb_params1['max_depth'] = 14

# lgb_params1['save_binary'] = True

# lgb_params1['metric'] = 'auc'

# lgb_params1['is_unbalance'] = True

# lgb_params1['boost_from_average'] = False

# lgb_params1['n_jobs'] =-1

# lgb_params1['verbose']=0

# lgb_params1['learning_rate'] = 0.24574985454914924

# lgb_params1['num_leaves'] = 2

# lgb_params1['max_bin'] = 300

# lgb_params1['min_sum_hessian_in_leaf'] = 1e-6

# lgb_params1['n_estimators'] = 1370

# lgb_params1['subsample'] = 1.0   

# lgb_params1['min_samples_split'] = 100

# lgb_params1['min_samples_leaf'] = 1



# lgb_params2 = {}

# lgb_params2['boosting_type']= 'gbdt'

# lgb_params2['objective'] = 'binary'

# lgb_params2['bagging_fraction']= 0.55

# lgb_params2['max_depth'] = 14

# lgb_params2['save_binary'] = True

# lgb_params2['metric'] = 'auc'

# lgb_params2['is_unbalance'] = True

# lgb_params2['boost_from_average'] = False

# lgb_params2['n_jobs'] =-1

# lgb_params2['verbose']=0

# lgb_params2['learning_rate'] = 0.24574985454914924

# lgb_params2['num_leaves'] = 2

# lgb_params2['max_bin'] = 300

# lgb_params2['min_sum_hessian_in_leaf'] = 1e-6

# lgb_params2['n_estimators'] = 1500

# lgb_params2['subsample'] = 1.0   

# lgb_params2['min_samples_split'] = 2

# lgb_params2['min_samples_leaf'] = 1



# lgb_params3 = {}

# lgb_params3['boosting_type']= 'gbdt'

# lgb_params3['objective'] = 'binary'

# lgb_params3['bagging_fraction']= 0.55

# lgb_params3['max_depth'] = 14

# lgb_params3['save_binary'] = True

# lgb_params3['metric'] = 'auc'

# lgb_params3['is_unbalance'] = True

# lgb_params3['boost_from_average'] = False

# lgb_params3['n_jobs'] =-1

# lgb_params3['verbose']=0

# lgb_params3['learning_rate'] = 0.20384544323121773

# lgb_params3['num_leaves'] = 2

# lgb_params3['max_bin'] = 300

# lgb_params3['min_sum_hessian_in_leaf'] = 0.049760518879130244

# lgb_params3['n_estimators'] = 1286

# lgb_params3['subsample'] = 0.7166484127307354   

# lgb_params3['min_samples_split'] = 79

# lgb_params3['min_samples_leaf'] = 92



# lgb_params4 = {}

# lgb_params4['boosting_type']= 'gbdt'

# lgb_params4['objective'] = 'binary'

# lgb_params4['bagging_fraction']= 0.55

# lgb_params4['max_depth'] = 14

# lgb_params4['save_binary'] = True

# lgb_params4['metric'] = 'auc'

# lgb_params4['is_unbalance'] = True

# lgb_params4['boost_from_average'] = False

# lgb_params4['n_jobs'] =-1

# lgb_params4['verbose']= 0

# lgb_params4['learning_rate'] = 0.01

# lgb_params4['num_leaves'] = 6

# lgb_params4['max_bin'] = 63

# lgb_params4['min_sum_hessian_in_leaf'] = 0.000446

# lgb_params4['n_estimators'] = 1286

# lgb_params4['subsample'] = 0.7166484127307354   

# lgb_params4['min_samples_split'] = 79

# lgb_params4['min_samples_leaf'] = 90



# lgb_model = lgb.LGBMClassifier(**lgb_params1)

# lgb_model2 = lgb.LGBMClassifier(**lgb_params2)

# lgb_model3 = lgb.LGBMClassifier(**lgb_params3)

# lgb_model4 = lgb.LGBMClassifier(**lgb_params4)

# log_model = LogisticRegression(solver='liblinear')
# stack = Ensemble(n_splits=10,

#         stacker = log_model,

#         base_models = (lgb_model, lgb_model2, lgb_model3, lgb_model4))
# test_df = pd.read_csv(TEST_DF)

# test_df.head()
# X_test = test_df.drop(['ID_code'], axis=1)

# scaler.fit_transform(X_test.values)

# predictions = lgb_gbm.predict(X_test, num_iteration=lgb_gbm.best_iteration)
# predictions = stack.fit_predict(X, Y, X_test) 
# submission_df = pd.read_csv(SUB_DF)

# submission_df.target = predictions

# submission_df.to_csv('submission.csv', index=False)

# submission_df.head()