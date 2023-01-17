# Loading Libraries

import os

from time import time

import numpy as np

import pandas as pd

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC #SupportVectorClassifier

from sklearn.cross_validation import KFold;

from sklearn.metrics import accuracy_score
data_train = pd.read_csv('../input/preproc2_train.csv')

data_test = pd.read_csv('../input/preproc2_test.csv')
#Preparing data :

X = data_train.drop(['PassengerId','Survived'], axis=1)

X = X.values # creates an array

y = data_train['Survived']

y = y.values

X_test = data_test.drop(['PassengerId'], axis=1)

X_test = X_test.values
ntrain = X.shape[0]

ntest = X_test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 9 # set folds for out-of-fold prediction

kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        print(self.clf.fit(x,y).feature_importances_)
def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
from scipy.stats import randint

from sklearn.model_selection import RandomizedSearchCV

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):

    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,

                                  n_jobs=-1, n_iter=nbr_iter, cv=9)

    #CV = Cross-Validation ( here using Stratified KFold CV)

    start = time()

    rdmsearch.fit(X,y)

    print('hyper-tuning time : %d seconds' % (time()-start))

    start = 0

    ht_params = rdmsearch.best_params_

    ht_score = rdmsearch.best_score_

    return ht_params, ht_score

    
est = RandomForestClassifier(n_jobs=-1, n_estimators=500)

rf_p_dist={'max_depth':[3,5,10,None],

              'max_features':randint(1,6),

               'criterion':['gini','entropy'],

               'bootstrap':[True,False],

               'min_samples_leaf':randint(1,10)

              }

rf_parameters, rf_ht_score = hypertuning_rscv(est, rf_p_dist, 40, X, y)

rf_parameters['n_jobs']=-1

rf_parameters['n_estimators']=500

print(rf_parameters)

print('Hyper-tuned model score :')

print(rf_ht_score)
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_parameters)

# Now we train our model and get our out-of-bag sets

rf_oof_train, rf_oof_test = get_oof(rf, X, y, X_test) 

# basic accuracy_score : 

print(accuracy_score(rf_oof_train.ravel(), y)*100)
est = ExtraTreesClassifier(n_jobs=-1, n_estimators=500)

et_p_dist = {'criterion':['gini','entropy'], 

             'max_features':randint(1,6), 

             'max_depth':[3,10,None],

             'bootstrap':[True,False],

             'min_samples_leaf':randint(1,10)

             }
et_parameters, et_ht_score = hypertuning_rscv(est, et_p_dist, 30, X, y)

et_parameters['n_jobs']=-1

et_parameters['n_estimators']=500

print(et_parameters)

print('Hyper-tuned model score :')

print(et_ht_score)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_parameters)

et_oof_train, et_oof_test = get_oof(et, X, y, X_test) 

print(accuracy_score(et_oof_train.ravel(), y)*100)
est = AdaBoostClassifier()

ada_p_dist={'learning_rate':[0.25,0.5,0.75,1.],

            'n_estimators':[100,250,500,650],

            }
ada_parameters, ada_ht_score = hypertuning_rscv(est, ada_p_dist, 10, X, y)

print(ada_parameters)

print('Hyper-tuned model score :')

print(ada_ht_score*100)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_parameters)

ada_oof_train, ada_oof_test = get_oof(ada, X, y, X_test) 

print(accuracy_score(ada_oof_train.ravel(), y)*100)
est = GradientBoostingClassifier()

gb_p_dist={'n_estimators':[100,250,500,750],

           'max_depth':[3,5,10,None],

           'min_samples_leaf':randint(1,10),

           }
gb_parameters, gb_ht_score = hypertuning_rscv(est, gb_p_dist, 40, X, y)

print(gb_parameters)

print('Hyper-tuned model score :')

print(gb_ht_score*100)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_parameters)

gb_oof_train, gb_oof_test = get_oof(gb, X, y, X_test) 

print(accuracy_score(gb_oof_train.ravel(), y)*100)
est = SVC()

from scipy.stats import norm

svc_p_dist={'kernel':['linear','poly','rbf'],

            'C':norm(loc=0.5, scale=0.15)} # A ABSOLUMENT REVOIR 
svc_parameters, svc_ht_score = hypertuning_rscv(est, svc_p_dist, 200, X, y)

print(svc_parameters)

print('Hyper-tuned model score :')

print(svc_ht_score*100)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_parameters)

svc_oof_train, svc_oof_test = get_oof(svc, X, y, X_test) 

print(accuracy_score(svc_oof_train.ravel(), y)*100)
Flevel_pred_train = pd.DataFrame({'RF': rf_oof_train.ravel(),

                                 'ET':et_oof_train.ravel(),

                                 'AB':ada_oof_train.ravel(),

                                 'GB':gb_oof_train.ravel(),

                                 'SVC':svc_oof_train.ravel()})

Flevel_pred_train.head()
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)

x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

 n_estimators= 2000,

 max_depth= 4,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1).fit(X, y)

predictions = gbm.predict(X_test)

print(accuracy_score(gbm.predict(X),y)*100)
PassengerId = data_test['PassengerId']

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)