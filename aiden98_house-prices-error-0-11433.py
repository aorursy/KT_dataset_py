#import some necessary librairies

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLars,RidgeCV

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
import pandas as pd

train =pd.read_csv('../input/housingames/X_train.csv')

test = pd.read_csv('../input/housingames/X_test.csv')

ytrain=pd.read_csv('../input/housingames/y_train.csv')
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9,

                                                random_state=7))

#########################################################################

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



#########################################################################



GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

# library used for stacking 

!pip install vecstack
from vecstack import stacking



estimators = [KRR,GBoost,ENet]

X_train=train

y_train=ytrain

X_test=test

k=5



L_train_1, L_test_1=stacking(estimators,X_train,

         y_train, X_test,regression=True, 

         n_folds=k,mode='oof_pred',random_state=7, 

         verbose=2)
ENet2 = make_pipeline(RobustScaler(), ElasticNet(alpha=0.00055, l1_ratio=.45,

                                                random_state=7))

#########################################################################

KRR2 = KernelRidge(alpha=0.4, kernel='polynomial', degree=2, coef0=2.5)

#########################################################################

GBoost2 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01,

                                   max_depth=3, max_features='sqrt',

                                   min_samples_leaf=7, min_samples_split=10, 

                                   loss='huber', random_state =7)
#layer 2

estimatorsL2=[ENet2,KRR2,GBoost2]



L_train_2, L_test_2=stacking(estimatorsL2,L_train_1,

         y_train, L_test_1,regression=True, 

         n_folds=k,mode='oof_pred',random_state=7, 

         verbose=2)

#our estimator (hyper params have been found by randomized search)

ENet3=make_pipeline(RobustScaler(), ElasticNet(alpha=0.006, l1_ratio=0.0008,

                                                random_state=7))
#layer 3

L_train_3, L_test_3=stacking([ENet3],L_train_2,

         y_train, L_test_2,regression=True, 

         n_folds=k,mode='oof_pred',random_state=7, 

         verbose=1)



print(rmsle(y_train,L_train_3))
stack_pred=np.expm1(L_test_3).reshape(len(L_test_3),)



#traing predictions are in logged form 

#because the y_train is still in this form too

stack_train=L_train_3.reshape(len(L_train_3),)


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

#########################################################################

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train, y_train)

lgb_train_pred = model_lgb.predict(train)

lgb_pred = np.expm1(model_lgb.predict(test.values))

print(rmsle(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''



print('RMSLE score on train data:')

print(rmsle(y_train,stack_train*0.7 +xgb_train_pred*0.12+ lgb_train_pred*0.18  ))
stack_pred=stack_pred.reshape(1459,)

ensemble =stack_pred*0.7 +xgb_pred*0.12 + lgb_pred*0.18  
ensemble.shape
sub = pd.DataFrame()

sub['Id'] = range(1461,1461+1459)

sub['SalePrice'] = ensemble

sub.to_csv('submission.csv',index=False)

sub.head()