# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import warnings                   # To ignore the warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
cross_validation = StratifiedKFold(n_splits=5, shuffle=False)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score
# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/CE802_Ass_2018_Data.csv")
test=pd.read_csv("../input/CE802_Ass_2018_Test.csv")
train.head()
test.head()
train.describe()
train.isnull().sum()
train.dtypes
train.head()
train_data=train.drop('Class',axis=1)
train_data.head()
target=train['Class']
target.head()
train_data.shape, target.shape
def normalization(data):
    return (data - data.min())/(data.max() - data.min())
train_data = normalization(train_data)
train_data.head()
#modelling
grid_parameters={"C":[1.0,1.5,2.0,2.5], "penalty":["l1","l2"] }
clf= LogisticRegression()
clf = GridSearchCV(clf, param_grid=grid_parameters, cv=cross_validation,n_jobs=-1)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print("K-Fold scores :",score)
print("Cross validation Accuracy:",round(np.mean(score)*100, 2))
y_pred = cross_val_predict(clf, train_data, target, cv=5)
print("F1 score:",f1_score(target, y_pred,average='weighted'))
print("precision score:",precision_score(target, y_pred,average='weighted'))
print("recall score:",recall_score(target, y_pred,average='weighted'))
tn, fp, fn, tp = confusion_matrix(target, y_pred).ravel()
fp, tp, thresold = roc_curve(target, y_pred)
AUC = auc(fp, tp)
print("ROC Area under Curve:",AUC)
y_pred.shape
target.shape
clf
clf = DecisionTreeClassifier()
grid_parameters = {'max_depth': range(10,30,1),'min_samples_split' : range(10,50,2),'min_samples_leaf' : range(10, 30,2)
                 }
clf = GridSearchCV(clf, param_grid=grid_parameters, cv=cross_validation,n_jobs=-1)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print("K-Fold scores :",score)
print("Cross validation Accuracy:",round(np.mean(score)*100, 2))
y_pred = cross_val_predict(clf, train_data, target, cv=5)
print("F1 score:",f1_score(target, y_pred,average='weighted'))
print("precision score:",precision_score(target, y_pred,average='weighted'))
print("recall score:",recall_score(target, y_pred,average='weighted'))
tn, fp, fn, tp = confusion_matrix(target, y_pred).ravel()
fp, tp, thresold = roc_curve(target, y_pred)
AUC = auc(fp, tp)
print("ROC Area under Curve:",AUC)
clf
clf = RandomForestClassifier(n_estimators=13)
param_grid = {"max_depth": [5, None],
              "max_features": [3,5,10],
              "min_samples_split": [2, 5, 7],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
clf = GridSearchCV(clf, param_grid=grid_parameters, cv=cross_validation,n_jobs=-1)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print("K-Fold scores :",score)
print("Cross validation Accuracy:",round(np.mean(score)*100, 2))
y_pred = cross_val_predict(clf, train_data, target, cv=5)
print("F1 score:",f1_score(target, y_pred,average='weighted'))
print("precision score:",precision_score(target, y_pred,average='weighted'))
print("recall score:",recall_score(target, y_pred,average='weighted'))
tn, fp, fn, tp = confusion_matrix(target, y_pred).ravel()
fp, tp, thresold = roc_curve(target, y_pred)
AUC = auc(fp, tp)
print("ROC Area under Curve:",AUC)
clf
clf = SVC()
grid_parameters = {   
                  'C': np.logspace(-1, 3, 9),
                 'gamma' : np.logspace(-7, -0, 8),
                 'kernel' :['rbf', 'linear']
                 }
clf = GridSearchCV(clf, param_grid=grid_parameters, cv=cross_validation,n_jobs=-1)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print("K-Fold scores :",score)
print("Cross validation Accuracy:",round(np.mean(score)*100, 2))
y_pred = cross_val_predict(clf, train_data, target, cv=5)
print("F1 score:",f1_score(target, y_pred,average='weighted'))
print("precision score:",precision_score(target, y_pred,average='weighted'))
print("recall score:",recall_score(target, y_pred,average='weighted'))
tn, fp, fn, tp = confusion_matrix(target, y_pred).ravel()
fp, tp, thresold = roc_curve(target, y_pred)
AUC = auc(fp, tp)
print("ROC Area under Curve:",AUC)
clf
test.head()
test_data=test.drop('Class',axis=1)
test=test.drop('Class',axis=1)
test.head()
test_data.head()
test_data = normalization(test_data)
test_data.head()
clf = SVC()
grid_parameters = {   
                  'C': np.logspace(-1, 3, 9),
                 'gamma' : np.logspace(-7, -0, 8),
                 'kernel' :['rbf', 'linear']
                 }
clf = GridSearchCV(clf, param_grid=grid_parameters, cv=cross_validation,n_jobs=-1)
clf.fit(train_data, target)
prediction = clf.predict(test_data)
test["Class"] = pd.DataFrame({"Class": prediction
})
test.to_csv('test.csv', index=False)
test
