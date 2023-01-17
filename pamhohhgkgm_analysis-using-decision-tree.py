import pandas as pd

import numpy as np

from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

data=load_iris() #we can directly load the data from Sklearn,as it already have it.  

print('Classes to predict: ', data.target_names)
data
X=data.data

y=data.target

print('Number of examples in the data:', X.shape[0])
X[:5] # upper 5 rows
X_train, X_test, y_train,y_test=train_test_split(X,y,random_state=47,test_size=0.30)
from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score

print('Accuracy score on the train data: ',accuracy_score(y_true=y_train,y_pred=clf.predict(X_train)))

print('Accuracy score on the test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))
clf=DecisionTreeClassifier(criterion='entropy',min_samples_split=40)

clf.fit(X_train, y_train)

print('Accuracy score on the train data: ', accuracy_score(y_true=y_train,y_pred=clf.predict(X_train)))

print('Accuracy score on the test data: ', accuracy_score(y_true=y_test,y_pred=clf.predict(X_test)))