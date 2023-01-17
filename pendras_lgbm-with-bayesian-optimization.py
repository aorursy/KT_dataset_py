import numpy as np

import pandas as pd

import os
app_train = pd.read_csv('../input/home-credit-simple-featuers/simple_features_train.csv')

app_test = pd.read_csv('../input/home-credit-simple-featuers/simple_features_test.csv')
sk_id = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
train = app_train.drop(columns = ['TARGET'])

train_labels = app_train['TARGET']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

train, train_labels, test_size=0.2)
from lightgbm import LGBMClassifier





clf = LGBMClassifier(nthread=4,n_estimators=10000,learning_rate=0.02,num_leaves=34,colsample_bytree=0.9497036,subsample=0.8715623,max_depth=8,reg_alpha=0.041545473,reg_lambda=0.0735294,min_split_gain=0.0222415,min_child_weight=39.3259775,silent=-1,verbose=-1 )







#clf = LGBMClassifier(boosting_type='gbdt', class_weight=None,colsample_bytree=0.6149378064887835, is_unbalance=True,learning_rate=0.0126346500398102, max_depth=-1, metric='auc',min_child_samples=390, min_child_weight=0.001, min_split_gain=0.0,

 #  n_estimators=1327, n_jobs=-1, num_leaves=106, objective=None,random_state=None, reg_alpha=0.5129992714397862, reg_lambda=0.38268769901820565, silent=True, subsample=0.7177561548329953, subsample_for_bin=80000,

  #     subsample_freq=0, verbose=1)

#clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_train, y_train)], 

            #eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)

    



clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], 

            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

    
#from sklearn.metrics import roc_auc_score

#y_pred = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1]

#roc_auc_score(y_test, y_pred)
y_pred = clf.predict_proba(app_test, num_iteration=clf.best_iteration_)[:, 1]



submit = sk_id[['SK_ID_CURR']]

submit['TARGET'] = y_pred



# Save the submission dataframe

submit.to_csv('sub1.csv', index = False)
