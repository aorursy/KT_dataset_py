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
from sklearn import datasets

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.metrics import classification_report
iris = datasets.load_iris()
from sklearn.model_selection import train_test_split



# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3,random_state=109)
param_grid = {'kernel':('sigmoid', 'rbf', 'linear'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}

#param_grid = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
clf = SVC(kernel='sigmoid') # Linear Kernel



#Train the model using the training sets

clf.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)

from sklearn import metrics



# Model Accuracy: how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

gd_sr = GridSearchCV(estimator=clf,  

                     param_grid=param_grid,

                     scoring='accuracy',

                     cv=5,

                     n_jobs=-1)

gd_sr.fit(X_train, y_train)
best_parameters = gd_sr.best_params_  

print(best_parameters)
#'C': 1, 'decision_function_shape': 'ovo', 'gamma': 1, 'kernel': 'linear', 'shrinking': True}

clf = SVC(C=1, decision_function_shape='ovo', gamma=1, kernel='linear', shrinking=True) # Linear Kernel



#Train the model using the training sets

clf.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

param_grid = {'n_estimators':(1, 5, 10, 50), 'criterion':('gini', 'entropy'), 'bootstrap':(True, False), 

              'min_samples_split':(2,3,5), 'random_state':(1,2)}



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=1)



clf.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
gd_sr = GridSearchCV(estimator=clf,  

                     param_grid=param_grid,

                     scoring='accuracy',

                     cv=5,

                     n_jobs=-1)

gd_sr.fit(X_train, y_train)

best_parameters = gd_sr.best_params_  

print(best_parameters)
clf = RandomForestClassifier(bootstrap = True, criterion='gini', min_samples_split=3, n_estimators=5, random_state=1)


clf.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))