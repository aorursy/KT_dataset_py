# Importing the libraries

import numpy as np 

import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dataset = pd.read_csv('../input/IRIS.csv')
pandas_profiling.ProfileReport(dataset)
#pairplot

sns.set_style("whitegrid");

sns.pairplot(dataset, hue="species", size=3);

# Importing metrics for evaluation

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =452 )
#importing libraries

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



# Summary of the predictions made by the classifier

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# Accuracy score

from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_pred,y_test))
from sklearn.svm import SVC



classifier = SVC()

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



# Summary of the predictions made by the classifier

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# Accuracy score

from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_pred,y_test))
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



# Summary of the predictions made by the classifier

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# Accuracy score

from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_pred,y_test))
from sklearn.tree import DecisionTreeClassifier



classifier = DecisionTreeClassifier()



classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



# Summary of the predictions made by the classifier

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# Accuracy score

from sklearn.metrics import accuracy_score

print('accuracy is',accuracy_score(y_pred,y_test))