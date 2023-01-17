# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.chdir("../input"))

# Any results you write to the current directory are saved as output.
os.listdir()
test=pd.read_csv('../input/test_set.csv')
train=pd.read_csv('../input/train_set.csv')
test.head()
train.head()
test.shape,train.shape
(train.isnull()).sum(axis=0)
len(train.columns[train.dtypes == 'float'])
#int=73, object=1,float=28
train.columns[-1]
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X_train=train.drop(['Customer_ID','Class'],axis=1)
Y_train=train['Class']
X_train.head()
Y_train.head()
X_train,X_test,Y_train,Y_test=train_test_split(X_train,Y_train,test_size=0.3)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)
pred=model.predict(X_test)
model.score(X_test, Y_test)
print((cross_val_score(model,X=X_test,y=Y_test,scoring='roc_auc')).mean())

from sklearn.metrics import accuracy_score
accuracy_score(y_true=Y_test,y_pred=pred)
rf = RandomForestClassifier()
params = {'n_estimators':[100,200,300],
          'criterion':['gini','entropy'],
          'max_depth':[5,7,9],
          'max_features':[3,5,7,27]}
from sklearn.model_selection import GridSearchCV
best_RF = GridSearchCV(estimator=rf, param_grid=params, 
                       scoring='roc_auc', cv =5, n_jobs=4)
import time
%timeit

best_RF.fit(X=X_train, y=Y_train)
best_RF.best_params_
RF=RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=7,max_features=27)
RF.fit(X_train,Y_train)
(cross_val_score(RF,X=X_train,y=Y_train,scoring='roc_auc')).mean()
pd=RF.predict(X_train)
RF.score(X_train,Y_train)
accuracy_score(y_true=Y_train,y_pred=pd)
pred=RF.predict(X_test)
RF.score(X_test,Y_test)
RF.predict
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
sample_submission = pd.DataFrame({'1':test.Customer_ID,'2':pd.Series(pred)})
sample_submission.columns= ['Customer_ID','Class']
sample_submission.to_csv('Submission_file.csv')
