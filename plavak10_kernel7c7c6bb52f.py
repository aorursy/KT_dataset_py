import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset=pd.read_csv("../input/credit-card-risk-assessment/Credit_default_dataset.csv")

dataset.head()
dataset=dataset.drop(['ID'],axis=1)

dataset.rename(columns={'PAY_0':'PAY_1'},inplace=True)

dataset.head()
dataset['EDUCATION'].value_counts()
dataset['MARRIAGE'].value_counts()
dataset['EDUCATION']=dataset['EDUCATION'].map({0:4,1:1,2:2,3:3,4:4,5:4,6:4})

dataset['MARRIAGE']=dataset['MARRIAGE'].map({0:3,1:1,2:2,3:3})
from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

X=dataset.drop(['default.payment.next.month'],axis=1)

X=scale.fit_transform(X)

y=dataset['default.payment.next.month']
parameters={

    "learning_rate":[0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],

    "min_child_weight":[ 1, 3, 5, 7 ],

    "max_depth":[ 3, 4, 5, 6, 8, 10, 12, 15],

    "gamma":[ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

    "colsample_bytree":[ 0.3, 0.4, 0.5 , 0.7 ]

    }
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import xgboost

classifier=xgboost.XGBClassifier()

def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))



random_search=RandomizedSearchCV(classifier,param_distributions=parameters,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
from datetime import datetime

start_time = timer(None)

random_search.fit(X,y)

timer(start_time)
random_search.best_estimator_
random_search.best_params_
classifier=xgboost.XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.4, gamma=0.2, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.05, max_delta_step=0, max_depth=6,

              min_child_weight=3, missing=None, monotone_constraints=None,

              n_estimators=100, n_jobs=0, num_parallel_tree=1,

              objective='binary:logistic', random_state=0, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

              validate_parameters=False, verbosity=None)
from sklearn.model_selection import cross_val_score

score=cross_val_score(classifier,X,y,cv=10)

score
score.mean()