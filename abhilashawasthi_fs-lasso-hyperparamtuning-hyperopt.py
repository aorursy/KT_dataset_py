# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import gc
import time
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval 
from hyperopt.pyll.base import scope
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['log_target'] = np.log1p(train.target)
print(train.shape, test.shape)
train.head(2)
X_cols = [col for col in train.columns if col not in ['ID','target','log_target']]
print(len(X_cols))
# Converting features into log transformation
feats_to_convert = []
for col in X_cols:
    diff = train[col].max() - train[col].min()
    if diff>1000:
        feats_to_convert.append(col)
print(len(feats_to_convert))

train[feats_to_convert] = np.log1p(train[feats_to_convert].values)
test[feats_to_convert] = np.log1p(test[feats_to_convert].values)
lasso_mod = Lasso(alpha=0.05, max_iter=1000, fit_intercept=True, normalize=False, random_state=42)
lasso_mod.fit(X=train[X_cols].values, y=train.log_target.values)
imp_feats_indexes = np.nonzero(lasso_mod.coef_)[0]
print(imp_feats_indexes)
imp_feats = np.array(X_cols)[imp_feats_indexes]
print('Number of important features selected by lasso:', len(imp_feats))
print('Important features are:', imp_feats)
def model_metrics(y_test, preds, scores):
    scores['rmsle'].append(np.sqrt(mean_squared_error(y_test, preds)))
    #scores['mae'].append(mean_absolute_error(y_test, preds))
    return scores

def get_space(clf_choice):
    if clf_choice=='LGB':
        lgb_space ={'num_leaves': scope.int(hp.quniform('num_leaves', 50, 200, 1)),
                    'learning_rate': 0.1, #hp.uniform('learning_rate', 0.02, 0.05),
                    'max_bin': scope.int(hp.quniform('max_bin', 300, 500, 1)),
                    'num_boost_round': scope.int(hp.quniform('num_boost_round', 100, 2000, 1)),
                    'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
                    'min_child_samples': scope.int(hp.quniform('min_child_samples', 1, 100, 1)), 
                    'subsample': hp.uniform('subsample', 0.5, 1.0), 
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0)}
        return lgb_space
    
def lgb_train(space, X_train, y_train, X_test):
    lgb_params ={'task':'train', 'boosting_type':'gbdt', 'objective':'regression', 'metric': {'rmse'},
                 'num_leaves': space['num_leaves'], 'learning_rate': space['learning_rate'], 'max_bin': space['max_bin'], 
                 'max_depth': space['max_depth'], 'min_child_samples':space['min_child_samples'], 'subsample': space['subsample'],
                 'colsample_bytree': space['colsample_bytree'], 'nthread':4, 'verbose': 0}
    lgbtrain = lgb.Dataset(X_train, label=y_train)
    lgbtrain.construct()
    lgb_model = lgb.train(lgb_params, lgbtrain, num_boost_round=space['num_boost_round'])
    preds = lgb_model.predict(X_test, num_iteration=space['num_boost_round'])
    return lgb_model, preds

def hyperopt_param_tuning(space, kf, clf_choice, trainX, trainY, max_evals):
    
    def objective(space):
        print('Space:', space)
        scores = {'rmsle':[]} #, 'mae':[]}
        for train_index, test_index in kf.split(trainX):
            X_train, X_test, y_train, y_test = trainX[train_index], trainX[test_index], trainY[train_index], trainY[test_index]
            lgb_model, preds = lgb_train(space, X_train, y_train, X_test)
            scores = model_metrics(y_test, preds, scores)
            print('scores', scores)
        loss = np.array(scores['rmsle']).mean()
        print('RMSLE:', loss, '\n\n')
        return{'loss':loss, 'status': STATUS_OK, 'scores':scores}
    
    trials = Trials()
    # Run the hyperparameter search
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    # Get the values of the optimal parameters
    best_params = space_eval(space, best)
    return best_params, trials

trainX = train[imp_feats].values
trainY = train['log_target'].values
# 5-fold CV
kf = KFold(n_splits=10, random_state=42)    
lgb_space = get_space('LGB')
# Hyperparameter tuning
lgb_best_params, trials = hyperopt_param_tuning(lgb_space, kf, 'LGB', trainX, trainY, 100)
print('Best LGB params for our task:', lgb_best_params)
#print(trials)
# Train on whole data using best params that we got from HyperOpt and lower learning_rate     
testX = test[imp_feats].values
#lgb_best_params['learning_rate'] = 0.001
#lgb_best_params['num_boost_round'] = lgb_best_params['num_boost_round'] * 80 
lgb_model, preds = lgb_train(lgb_best_params, trainX, trainY, testX)
sub = test[['ID']]
sub['target'] = np.expm1(preds)
print(sub.head(), '\n')
print(sub.describe())
sub.to_csv('first_model_prediction.csv', index=False)
