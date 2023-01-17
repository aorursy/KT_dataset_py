# importing important libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set(color_codes = True)
# loading dataset

dataset = pd.read_csv("../input/Iris.csv")
dataset.head()
# droping Id column

dataset = dataset.drop('Id',axis=1)
print(dataset.info())
# getting feel of data

print(dataset.describe())

# gives mea, standar deviation etc of data in different columns
# getting feel of data contd..

print(dataset.groupby("Species").size())
sns.pairplot(dataset, hue="Species",diag_kind="kde")
sns.violinplot(data = dataset, x="PetalLengthCm", y="Species")
sns.violinplot(data = dataset, x="SepalWidthCm", y="Species")
sns.violinplot(data = dataset, x="PetalWidthCm", y="Species")
# we can see that Iris setosa can be easily differentiated from iris versicolor and iris virginica,
# the latter two are intercepting each other and are difficult to differentiate. Now we will see if
# ML algorithms can differentiate other two 
# importing cross_validation functions

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# preparing data 

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting data to train and test data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Training on linear model

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
# Naive Bayes

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
# training on support vector machine

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
# K-Nearest Neighbours

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
# Decision Tree's

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
# as can be seen linear model and naive bayes have an accuracy of .967 which is pretty good. 
# But SVM, Decision tree and K nearest neighbor have an accuracy of 1.0 meaning all test data 
# were accurately determind by the model