# Importing all classifiers

import pandas as pd

import numpy

import os

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
inputText_train = pd.read_csv("../input/titanic-training-dataset/train.csv")

inputText_test = pd.read_csv("../input/titanic-test-data/test.csv")
inputText_train_cleaned = inputText_train.drop(columns=['PassengerId','Name','Ticket','Cabin'], axis=1)

inputText_train_cleaned.replace('',numpy.nan,inplace=True)

inputText_train_cleaned.fillna(inputText_train_cleaned.mean(), inplace=True)
inputText_test_cleaned = inputText_test.drop(columns=['PassengerId','Name','Ticket','Cabin'], axis=1)

inputText_test_cleaned.replace('',numpy.nan,inplace=True)

inputText_test_cleaned.fillna(inputText_test_cleaned.mean(), inplace=True)
featurelist_categorical = ['Pclass', 'Sex', 'Embarked','SibSp','Parch']

inputText_train_with_dummies = pd.get_dummies(inputText_train_cleaned,columns=featurelist_categorical)

inputText_test_with_dummies = pd.get_dummies(inputText_test_cleaned,columns=featurelist_categorical)
X = inputText_train_with_dummies.drop(columns=['Survived'], axis=1)

y_true = inputText_train['Survived']
#model = RandomForestClassifier(n_estimators=100)

#model = MultinomialNB()

#model = LogisticRegression()

#model = DecisionTreeClassifier()

#model = KNeighborsClassifier()

model = XGBClassifier()

#model = RandomForestClassifier())





model.fit(X, y_true)

y_predicted = model.predict(inputText_train_with_dummies.drop(columns=['Survived'], axis=1))

y_predicted_test = model.predict(inputText_test_with_dummies)



print(y_predicted_test)