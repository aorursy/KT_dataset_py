import os

import numpy as np

import pandas as pd

from sklearn import preprocessing

import xgboost as xgb

import lightgbm as lgb

import optuna

import functools

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve,auc,accuracy_score,confusion_matrix,f1_score
train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')



train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')



sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')



train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print(train.shape)

print(test.shape)



y_train = train['isFraud'].copy()

del train_transaction, train_identity, test_transaction, test_identity



# Drop target, fill in NaNs

X_train = train.drop('isFraud', axis=1)

X_test = test.copy()



del train, test



X_train = X_train.fillna(-999)

X_test = X_test.fillna(-999)



# Label Encoding

for f in X_train.columns:

    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train[f].values) + list(X_test[f].values))

        X_train[f] = lbl.transform(list(X_train[f].values))

        X_test[f] = lbl.transform(list(X_test[f].values))   
(X_train,X_eval,y_train,y_eval) = train_test_split(X_train,y_train,test_size=0.2,random_state=0)
def opt(X_train, y_train, X_test, y_test, trial):

    #param_list

    n_estimators = trial.suggest_int('n_estimators', 0, 1000)

    max_depth = trial.suggest_int('max_depth', 1, 20)

    min_child_weight = trial.suggest_int('min_child_weight', 1, 20)

    #learning_rate = trial.suggest_discrete_uniform('learning_rate', 0.01, 0.1, 0.01)

    scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 100)

    subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)

    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)



    xgboost_tuna = xgb.XGBClassifier(

        random_state=42, 

        tree_method='gpu_hist',

        n_estimators = n_estimators,

        max_depth = max_depth,

        min_child_weight = min_child_weight,

        #learning_rate = learning_rate,

        scale_pos_weight = scale_pos_weight,

        subsample = subsample,

        colsample_bytree = colsample_bytree,

    )

    xgboost_tuna.fit(X_train, y_train)

    tuna_pred_test = xgboost_tuna.predict(X_test)

    

    return (1.0 - (accuracy_score(y_test, tuna_pred_test)))
study = optuna.create_study()

study.optimize(functools.partial(opt, X_train, y_train, X_eval, y_eval), n_trials=100)
study.best_params
clf = xgb.XGBClassifier(tree_method='gpu_hist',**study.best_params)

clf.fit(X_train, y_train)
sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]

sample_submission.to_csv('submission.csv')