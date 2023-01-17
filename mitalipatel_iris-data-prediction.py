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
import matplotlib.pyplot as plt
data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

data.head()
X = data.iloc[:,:-1].values

y = data.iloc[:,4].values
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
from sklearn.svm import SVC



svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

svm.fit(X_train, y_train)

print('The accuracy of the SVM classifier on training data is {:.2f}'.format(svm.score(X_train, y_train)))

print('The accuracy of the SVM classifier on test data is {:.2f}'.format(svm.score(X_test, y_test)))
#Applying XGBoost

import xgboost as xgb



xgb_clf = xgb.XGBClassifier()

xgb_clf = xgb_clf.fit(X_train, y_train)



print('The accuracy of the XGBoost classifier on training data is {:.2f}'.format(xgb_clf.score(X_train, y_train)))

print('The accuracy of the XGBoost classifier on test data is {:.2f}'.format(xgb_clf.score(X_test, y_test)))
#Applying Decision Tree

from sklearn import tree



#Create tree object

decision_tree = tree.DecisionTreeClassifier(criterion='gini')



#Train DT based on scaled training set

decision_tree.fit(X_train, y_train)



#Print performance

print('The accuracy of the Decision Tree classifier on training data is {:.2f}'.format(decision_tree.score(X_train, y_train)))

print('The accuracy of the Decision Tree classifier on test data is {:.2f}'.format(decision_tree.score(X_test, y_test)))
#Applying RandomForest

from sklearn.ensemble import RandomForestClassifier



#Create Random Forest object

random_forest = RandomForestClassifier()



#Train model

random_forest.fit(X_train, y_train)



#Print performance

print('The accuracy of the Random Forest classifier on training data is {:.2f}'.format(random_forest.score(X_train, y_train)))

print('The accuracy of the Random Forest classifier on test data is {:.2f}'.format(random_forest.score(X_test, y_test)))