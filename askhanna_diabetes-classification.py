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
diabetes = pd.read_csv('../input/diabetes.csv')
diabetes.head()
import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure(figsize=(16,10))
#ax = fig.add_subplot([1,2])
ax = fig.add_subplot(121)
ax = plt.hist(diabetes[diabetes['Outcome']==1]['BloodPressure'])
ax2 = fig.add_subplot(122)
ax2 = plt.hist(diabetes[diabetes['Outcome']==0]['BloodPressure'])
y = diabetes['Outcome']
diabetes.columns
norm_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
diabetes[norm_cols] = diabetes[norm_cols].apply(lambda x: (x - x.min())/(x.max()-x.min()))
X = diabetes[norm_cols]
print(X.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.svm import SVC
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
clf_logi = LogisticRegression()
clf_logi.fit(X_train,y_train)
y_pred = clf_logi.predict(X_test)
print(accuracy_score(y_true=y_test,y_pred=y_pred))
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
print(cm)
clf_sv = SVC(kernel="linear", C=0.1)
clf_sv.fit(X_train,y_train)
y_pred = clf_sv.predict(X_test)
print(accuracy_score(y_true=y_test,y_pred=y_pred))
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
print(cm)
from sklearn.model_selection import cross_val_score
score = cross_val_score(clf_sv,X_train,y_train,cv=5)
print(score.mean())
y_pred = clf_sv.predict(X_test)
print(accuracy_score(y_true=y_test,y_pred=y_pred))
cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
print(cm)
from sklearn.grid_search import GridSearchCV

# prepare a range of values to test
param_grid = [
  {'C': [.01, .1, 1, 10], 'kernel': ['linear']},
 ]

grid = GridSearchCV(param_grid=param_grid,estimator = clf_sv)
grid.fit(X_train, y_train)
print(grid)
# summarize the results of the grid search
print("Best score: %0.2f%%" % (100*grid.best_score_))
print("Best estimator for parameter C: %f" % (grid.best_estimator_.C))
clf_sv = SVC(kernel="linear", C=0.1)
clf_sv.fit(X_train, y_train)
y_eval = clf_sv.predict(X_test)
acc = sum(y_eval == y_test) / float(len(y_test))
print("Accuracy: %.2f%%" % (100*acc))
diabetes[diabetes['Outcome']==1].plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))