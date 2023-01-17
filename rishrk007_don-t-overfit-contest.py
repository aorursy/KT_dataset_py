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
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.shape
test.shape
train.head(10)
train.isnull().sum()
train.pop("id")
test.pop("id")
y_train=train.iloc[:,0]
y_train
train.pop("target")
train.shape
train1=train.copy()

y_train1=y_train.copy()

test1=test.copy()
train.head(10)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train=sc.fit_transform(train)
test=sc.fit_transform(test)
from sklearn.linear_model import LogisticRegression,Lasso , Ridge

from sklearn.model_selection import GridSearchCV, cross_val_score

random_state=42

log_clf=LogisticRegression(random_state=random_state)

param_grid={ 

                'class_weight': ['balanced', None],

                  'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],

                  'penalty':['l1','l2']

           }

grid=GridSearchCV(log_clf,param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1)

grid.fit(train,y_train)



print("Best Score:" + str(grid.best_score_))

print("Best Parameters: " + str(grid.best_params_))



log_clf = LogisticRegression(C=0.1,random_state=42,class_weight='balanced',penalty='l1')
log_clf.fit(train,y_train)
pred_log=log_clf.predict(test)
pred_log #prediction by logistic regression
submission3 = pd.read_csv('../input/sample_submission.csv')
submission3["target"]=pred_log
submission3.to_csv('submissionlog.csv', index=False)
from sklearn.feature_selection import RFE

selector = RFE(log_clf, 25, step=1)

selector.fit(train,y_train)
submission1 = pd.read_csv('../input/sample_submission.csv')

submission1['target'] = selector.predict_proba(test) #prediction by rfe

submission1.to_csv('submissionRFE.csv', index=False)
from sklearn.ensemble import GradientBoostingRegressor
random_state=42

gbr=GradientBoostingRegressor(random_state=random_state)

param_grid={ 

     "learning_rate": [0.01, 0.05 , 0.1 , 1],

    "max_depth":[5, 11, 17, 25, 37, 53],

    "max_features":["log2","sqrt"],

    "criterion": ["friedman_mse",  "mae"],

    "n_estimators":[10, 50, 200, 600, 1000]

    }

           

grid=GridSearchCV(gbr,param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1)

grid.fit(train,y_train)



print("Best Score:" + str(grid.best_score_))

print("Best Parameters: " + str(grid.best_params_))



gbr=GradientBoostingRegressor( learning_rate=0.01, max_leaf_nodes=10, criterion='friedman_mse', max_depth=11, max_features='sqrt', n_estimators=1000)
gbr.fit(train,y_train)
pred5=gbr.predict(test)
pred5
submissionm = pd.read_csv('../input/sample_submission.csv')

submissionm["target"]=pred5
j=0

for i in submissionm["target"]:

    if i>1:

        submissionm["target"][j]=1

    j+=1    
submissionm.to_csv('submission_gbr.csv', index=False)
finalsub=(submissionm["target"]+submission1["target"])/2
finalsub
subf= pd.read_csv('../input/sample_submission.csv')

subf["target"]=finalsub
subf.to_csv('submission_fin.csv', index=False)