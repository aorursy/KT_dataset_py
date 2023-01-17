# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# import the data set

train=pd.read_csv('/kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv')
#view the data

train
#in  the data set the columns RowNumber,CustomerId,Surname are not mandatory,so skip them using iloc
X_train=train.iloc[:,3:13]
X_train


y_train=train.iloc[:,13:14]
y_train
#handle categorical data

geography=pd.get_dummies(X_train['Geography'],drop_first=True)
geography
gender=pd.get_dummies(X_train['Gender'],drop_first=True)
gender
# Drop Categorical Features

X_train=X_train.drop(['Geography','Gender'],axis=1)
X_train=pd.concat([X_train,geography,gender],axis=1)
## Hyper Parameter Optimization



params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

    

}
## Hyperparameter optimization using RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


import xgboost


classifier=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(X_train,y_train)
random_search.best_estimator_
# after getting the best parameters initialize the in the XGBClassifier as below 

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.7, gamma=0.3,

              learning_rate=0.1, max_delta_step=0, max_depth=8,

              min_child_weight=7, missing=None, n_estimators=100, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)
#validate the model performance using k fold cross validation

from sklearn.model_selection import cross_val_score

score=cross_val_score(classifier,X_train,y_train,cv=10)
score
score.mean()*100