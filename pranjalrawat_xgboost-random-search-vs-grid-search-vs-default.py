# This notebook compares Random Search vs Grid Search vs Default Parameters

# The algorithm of choice is XGBoost because:

# (a) it has sufficiently large number of parameters to tune (unlike say LogisticRegression, where there is really just one parameter to tune)

# (b) parameter tuning has, historically, shown to improve results drastically on XGBoost (unlike say Catboost, where tuning doesn't matter that much)

# The dataset has been artificially created to create a balanced data set with sufficient pattern and sufficient noise to bring out the comparision

# It can be theoretically shown that randomly selecting 60 parameter values from any parameter grid will ensure that there is 95% probability that

# at least 1 of those 60, lie within 5% area (with respect to total grid area) of the optimal parameter combination

# Testing used -> 5-fold cross validation used to prevent overfitting, 100 n_estimators kept as constant accross all comparisions
# generate random data - > only a few features are helpful, rest are noise

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000,

                           n_features=20,

                           n_informative=5,

                           n_redundant=0, 

                           class_sep =0.001,

                          random_state =2)
# balanced dataset

y.mean()
# preprocessing

import xgboost as xgb

dtrain = xgb.DMatrix(X,y)

model = xgb.XGBClassifier()

import time
# Case 1: Default Parameters for XGBoost



param = {'base_score': 0.5,

 'booster': 'gbtree',

 'colsample_bylevel': 1,

 'colsample_bytree': 1,

 'gamma': 0,

 'learning_rate': 0.1,

 'max_delta_step': 0,

 'max_depth': 3,

 'min_child_weight': 1,

 'missing': None,

 'n_estimators': 100,

 'n_jobs': 1,

 'nthread': 4,

 'objective': 'binary:logistic',

 'random_state': 0,

 'reg_alpha': 0,

 'reg_lambda': 1,

 'scale_pos_weight': 1,

 'silent': 0,

 'subsample': 1}

num_round = 100

start = time.time()

xgb.cv(param, dtrain, num_round, nfold=5, 

       metrics={'auc'}, seed=0,

       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

end = time.time()

print(end - start)
# Results on Default

print('Default Result')

print('Time Taken: 3.35 seconds')

print('Best AUC: 0.874242')
from sklearn.model_selection import GridSearchCV 



fixed_params = {

 'base_score': 0.5,

 'booster': 'gbtree',

 'colsample_bylevel': 1,

 'colsample_bytree': 1,

 'max_delta_step': 0,

 'missing': None,

 'n_estimators': 100,

 'n_jobs': 1,

 'nthread': 4,

 'objective': 'binary:logistic',

 'random_state': 0,

 'reg_alpha': 0,

 'reg_lambda': 1,

 'scale_pos_weight': 1,

 'silent': 1,

 'subsample': 1}



xgb_model = xgb.XGBClassifier(fixed_params)



grid_params = {

 'max_depth':[3,5,7],

 'learning_rate': [0.1, 0.05, 0.01],

 'reg_lambda': [0, 1, 3], 

 'gamma':[0,1,3],

 'min_child_weight': [1, 5, 10]

}



start = time.time()

GSmodel = GridSearchCV(estimator = xgb_model,scoring='roc_auc', param_grid = grid_params, cv = 5, verbose=2)

GSmodel.fit(X,y)

end = time.time()

print('Time Taken:', end - start)

print('Best Params:', GSmodel.best_params_) 

print('Best AUC:', GSmodel.best_score_)
# Random Search: With search in same parameter space as Grid Search



from sklearn.model_selection import RandomizedSearchCV 

from scipy.stats import randint as sp_randint

from scipy.stats import uniform 



fixed_params = {

 'base_score': 0.5,

 'booster': 'gbtree',

 'colsample_bylevel': 1,

 'colsample_bytree': 1,

 'max_delta_step': 0,

 'missing': None,

 'n_estimators': 100,

 'n_jobs': 1,

 'nthread': 4,

 'objective': 'binary:logistic',

 'random_state': 0,

 'reg_alpha': 0,

 'reg_lambda': 1,

 'scale_pos_weight': 1,

 'silent': 1,

 'subsample': 1}



xgb_model =xgb.XGBClassifier(fixed_params)



grid_params = {

 'max_depth':sp_randint(3, 7),

 'learning_rate': uniform(0.01, 0.11),

 'reg_lambda': uniform(1, 3),

 'gamma':uniform(0, 3),

 'min_child_weight': sp_randint(1, 10),

}



start = time.time()

RSmodel = RandomizedSearchCV(estimator = xgb_model,scoring='roc_auc', param_distributions = grid_params, cv = 5, verbose=2, n_iter = 60)

RSmodel.fit(X,y)

end = time.time()

print('Time Taken:', end - start)

print('Best Params:', RSmodel.best_params_) 

print('Best AUC:', RSmodel.best_score_)
# Random Search: With Wider Parameter Space for Search



fixed_params = {

 'base_score': 0.5,

 'booster': 'gbtree',

 'colsample_bylevel': 1,

 'colsample_bytree': 1,

 'max_delta_step': 0,

 'missing': None,

 'n_estimators': 100,

 'n_jobs': 1,

 'nthread': 4,

 'objective': 'binary:logistic',

 'random_state': 0,

 'reg_alpha': 0,

 'reg_lambda': 1,

 'scale_pos_weight': 1,

 'silent': 1,

 'subsample': 1}



xgb_model =xgb.XGBClassifier(fixed_params)



grid_params = {

 'max_depth':sp_randint(3, 12),

 'learning_rate': uniform(0.001, 0.101),

 'reg_lambda': uniform(1, 5),

 'gamma':uniform(0, 5),

 'min_child_weight': sp_randint(1, 10),

}



start = time.time()

RSmodel = RandomizedSearchCV(estimator = xgb_model,scoring='roc_auc', param_distributions = grid_params, cv = 5, verbose=2, n_iter = 60)

RSmodel.fit(X,y)

end = time.time()

print('Time Taken:', end - start)

print('Best Params:', RSmodel.best_params_) 

print('Best AUC:', RSmodel.best_score_)
# Results

# Default: 3.35 seconds, 0.87 AUC or 0.75 Gini

# Grid: 2196 seconds, 0.933 AUC or 0.86 Gini

# Random Search (With parameter Space as Grid Search): 462 seconds, 0.927 AUC or 0.86 Gini

# Random Search (With wider parameter Space than Grid Search): 735 seconds, 0.94 AUC or 0.87 Gini
# Verdict/Inference: given same time, RS will get better results than GS;

# given same parameter grid, RS will get similar results as GS in much lesser time. 

# Thus in practice: RS (Wider Search Space) > RS (Same Param Grid) > GS (Same Param Grid) > Default Settings