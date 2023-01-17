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
BT_train=pd.read_csv("/kaggle/input/scm-final-evaluation-challenge/new_BT_train.csv")

BT_test=pd.read_csv("/kaggle/input/scm-final-evaluation-challenge/new_BT_test.csv")

AT_train=pd.read_csv("/kaggle/input/scm-final-evaluation-challenge/new_AT_train.csv")

AT_test=pd.read_csv("/kaggle/input/scm-final-evaluation-challenge/new_AT_test.csv")
import pandas_profiling

AT_train.profile_report()
BT_train.profile_report()
TestFeature=AT_train

TestFeatureTest=AT_test
Distrib=AT_train

Distrib=Distrib.drop(['label'], axis='columns', inplace=False).unstack()

Distrib2=BT_train

Distrib2=Distrib2.drop(['label'], axis='columns', inplace=False).unstack()
# After training histogram

Distrib.hist()

#Before training histogram

Distrib2.hist()
# importing necessary libraries 

from sklearn import datasets 

from sklearn.metrics import confusion_matrix, classification_report 

from sklearn.model_selection import train_test_split
Y=TestFeature["label"]

data=TestFeature

X=data.drop(['label'], axis='columns', inplace=False)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y)
from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import cross_validate

dtree_model = DecisionTreeClassifier(criterion='entropy',splitter="best").fit(X_train, Y_train) 

dtree_predictions = dtree_model.predict(X_val) 

print("decision tree validation score :",dtree_model.score(X_val,Y_val))

print("decision tree training score :",dtree_model.score(X_train,Y_train))

y_true, y_pred = Y_val , dtree_predictions

print('Results on the test set:')

print(classification_report(y_true, y_pred))

cv_results = cross_validate(dtree_model, X_train, Y_train, cv=5)

np.mean(cv_results['test_score'])
! pip install -q scikit-plot
from sklearn.ensemble import AdaBoostClassifier

Ada_model = AdaBoostClassifier(dtree_model,n_estimators=100).fit(X_train, Y_train) 

Ada_predictions = Ada_model.predict(X_val) 

print("Adaboost validation score :",Ada_model.score(X_val,Y_val))

print("Adaboost training score :",Ada_model.score(X_train,Y_train))

y_true, y_pred = Y_val , Ada_predictions

print('Results on the test set:')

print(classification_report(y_true, y_pred))

cv_results = cross_validate(Ada_model, X_train, Y_train, cv=5)

print(np.mean(cv_results['test_score']))
import scikitplot as skplt



skplt.metrics.plot_confusion_matrix(

    Y_val, 

    Ada_predictions,normalize=True,

    figsize=(10,10))
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier 

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import normalize

from sklearn.neighbors import KNeighborsClassifier
RandomForest=RandomForestClassifier(criterion='entropy',n_estimators=200)

RandomForest.fit(X_train,Y_train)

RandomForest_predictions = RandomForest.predict(X_val)

print("RandomForest validation score:",RandomForest.score(X_val,Y_val))

print("RandomForest training score:",RandomForest.score(X_train,Y_train))

y_true, y_pred = Y_val , RandomForest.predict(X_val)

print('Results on the test set:')

print(classification_report(y_true, y_pred))

from sklearn.model_selection import cross_validate

cv_results = cross_validate(RandomForest, X_train, Y_train, cv=5)

print(np.mean(cv_results['test_score']))
#Import libraries:

import pandas as pd

import numpy as np

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import cross_val_score

from sklearn import metrics   #Additional scklearn functions

from sklearn.model_selection import GridSearchCV #Perforing grid search
xgb1 = XGBClassifier(objective= 'multi:softmax',probability=True,n_estimators=100,colsample_bytree=0.3,colsample_bylevel=0.7,max_depth=6)

xgb1.fit(X_train,Y_train)

print("XGB validation score:",xgb1.score(X_val,Y_val))

print("XGB training score:",xgb1.score(X_train,Y_train))

y_true, y_pred = Y_val , xgb1.predict(X_val)

print('Results on the test set:')

print(classification_report(y_true, y_pred))

cv_results = cross_validate(xgb1, X_train, Y_train, cv=5)

print(np.mean(cv_results['test_score']))

print(xgb1.get_params)
skplt.metrics.plot_confusion_matrix(

    Y_val, 

    xgb1.predict(X_val),normalize=True,

    figsize=(10,10))
from sklearn import svm

SVC= svm.SVC(probability=True,gamma='auto')

SVC.fit(X_train,Y_train)

y_true, y_pred = Y_val , SVC.predict(X_val)

from sklearn.metrics import classification_report

print('Results on the test set:')

print(classification_report(y_true, y_pred))

print("SVC validation score:",SVC.score(X_val,Y_val))

print("SVC training score:",SVC.score(X_train,Y_train))

cv_results = cross_validate(SVC, X_train, Y_train, cv=5)

print(np.mean(cv_results['test_score']))
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=7)

KNN.fit(X_train, Y_train)

y_true, y_pred = Y_val , KNN.predict(X_val)

print('Results on the test set:')

print(classification_report(y_true, y_pred))

print("KNN validation score:",KNN.score(X_val,Y_val))

print("KNN training score:",KNN.score(X_train,Y_train))

cv_results = cross_validate(KNN, X_train, Y_train, cv=5)

print(np.mean(cv_results['test_score']))
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_val_score

Vote = VotingClassifier(estimators=[('SVM', SVC),("xgb",xgb1),("Adaboost",Ada_model)],voting='soft')

for clf, label in zip([SVC,xgb1,Ada_model,Vote], ['SVM','XGB',"Adaboost",'vote']):

    scores = cross_val_score(clf, X_train, Y_train, scoring='accuracy', cv=5)

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
Vote.fit(X_train,Y_train)

y_true, y_pred = Y_val , Vote.predict(X_val)

print('Results on the test set:')

print(classification_report(y_true, y_pred))

from sklearn.model_selection import cross_validate

cv_results = cross_validate(KNN, X_train, Y_train, cv=5)

print("Vote validation score:",Vote.score(X_val,Y_val))

print("Vote training score:",Vote.score(X_train,Y_train))

print(np.mean(cv_results['test_score']))
skplt.metrics.plot_confusion_matrix(

    Y_val, 

    Vote.predict(X_val),normalize=True,

    figsize=(10,10))
resultat=Vote.predict(AT_test)

submission=pd.DataFrame()

submission['id']=range(320)

submission['label']=resultat
submission
filename = 'xgb.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)