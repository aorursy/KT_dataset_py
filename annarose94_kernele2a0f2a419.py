import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pylab as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
testTitanic = pd.read_csv('../input/test.csv')
trainTitanic = pd.read_csv('../input/train.csv')
testTitanic.head()
trainTitanic.head()
trainTitanic.info()
trainTitanic.describe(include=['O'])
#Fill the null values in age
trainTitanic.loc[trainTitanic.Age.isnull(), 'Age'] = trainTitanic.groupby('Pclass')['Age'].transform('mean')
testTitanic.loc[testTitanic.Age.isnull(), 'Age'] = testTitanic.groupby('Pclass')['Age'].transform('mean')
trainTitanic = trainTitanic.drop('Cabin', axis=1)
testTitanic = testTitanic.drop('Cabin', axis=1)
trainTitanic = trainTitanic.drop(['PassengerId','Name','Ticket'], axis=1)
testTitanic    = testTitanic.drop(['Name','Ticket'], axis=1)
trainTitanic["Embarked"] = trainTitanic["Embarked"].fillna("S")
testTitanic["Embarked"] = testTitanic["Embarked"].fillna("S")

features = ['Fare', 'Pclass', 'Sex']
trainTitanic = pd.get_dummies(trainTitanic,columns = ['Pclass', 'Sex'], drop_first = True)
trainTitanic.head()
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, t_test = train_test_split(trainTitanic.drop('Survived', axis = 1), trainTitanic['Survived'], test_size = 0.2)
#Logistic regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
trainTitanic.dtypes
model = RFE(logreg, 15)
model.fit(X_train, y_train)
#K Nearest Neighbors
model2 = KNeighborsClassifier()
model2.fit(X_train, y_train)
predictions = model2.predict(X_test)
score2 = (y_test, predictions)
#Decision Tree
model3=DecisionTreeClassifier()
model3.fit(X_train, y_train)
predictions = model3.predict(X_test)
score3(y_test, predicitions)