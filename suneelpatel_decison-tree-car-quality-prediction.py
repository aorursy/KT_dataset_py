import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv('../input/car.data', names=['buying','maint','doors','persons','lug_boot','safety','class'])
data.shape
data.head()
data.info()
sns.countplot('class', data= data)
data['buying'],_ = pd.factorize(data['buying'])

data['maint'],_ = pd.factorize(data['maint'])

data['doors'],_ = pd.factorize(data['doors'])

data['persons'],_ = pd.factorize(data['persons'])

data['lug_boot'],_ = pd.factorize(data['lug_boot'])

data['safety'],_ = pd.factorize(data['safety'])

data.head()
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)

y
y_inverse = le.inverse_transform([0,1,2,3])

y_inverse
sns.countplot(y)
# split data randomly into 70% training and 30% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape)

print(X_test.shape)

print(y_test.shape)

print(y_test.shape)
#train the decision tree

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)
# use the model to make predictions with the test data

y_pred = dtc.predict(X_test)
# What is the accuracy score?

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
# how did our model perform?

count_misclassified = (y_test != y_pred).sum()

count_misclassified
# How can we understand confusion matrix?

confusion_matrix(y_test, y_pred)
# What is the classification report?

print(classification_report(y_test, y_pred))