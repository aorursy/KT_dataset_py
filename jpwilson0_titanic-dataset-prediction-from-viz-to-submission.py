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
import seaborn as sns

import matplotlib.pyplot as plt
training_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
training_data
test_data
training_data.head()
training_data.tail()
training_data.describe()
training_data.info()
sns.countplot(training_data['Survived'])
sns.countplot(training_data['Sex'], hue = training_data['Survived'])
sns.countplot(training_data['Sex'], hue = training_data['Survived'])
sns.countplot(training_data['SibSp'], hue = training_data['Survived'])
sns.countplot(training_data['Parch'], hue = training_data['Survived'])
sns.countplot(training_data['Embarked'], hue = training_data['Survived'])
sns.countplot(training_data['Pclass'], hue = training_data['Survived'])
training_data['Age'].hist(bins = 20, color = 'g')
training_data['Fare'].hist(bins = 20, color = 'g')
training_data
training_data.isnull().any()
training_data.isnull().sum()
test_data.isnull().sum()
training_data.drop(['Cabin'], axis = 1, inplace=True)
training_data
training_data['Age'].median()
training_data['Age'].fillna(training_data['Age'].median(), inplace=True)
training_data.isna().any()
training_data.shape
training_data.dropna(subset = ['Embarked'], inplace=True)
training_data.shape
training_data.isna().sum()
embarked_dummies = pd.get_dummies(training_data['Embarked'])
training_data_ohe = pd.concat([training_data.drop(['Embarked'], axis =1), embarked_dummies], axis =1)

#Importing LabelEncoder from sklearn

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

training_data_ohe['Sex'] = le.fit_transform(training_data_ohe['Sex'])
training_data_ohe.head()
X = training_data_ohe.drop(['PassengerId', 'Name', 'Survived', 'Fare', 'Ticket'], axis = 1)
y = training_data_ohe['Survived']
print("We will be training the model using features that have:",X.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
print("The data for training contains: ", X_train.shape[0], "rows and ", X_train.shape[1], "columns.")

print("The data for testing contains: ", X_test.shape[0], "rows and ", X_test.shape[1], "columns.")
#Importing XGBoost

import xgboost
#Fitting XGBClassifier in the Training Data

xgb_classifier = xgboost.XGBClassifier()

xgb_classifier.fit(X_train, y_train)
# Let's make predictions in the X_test

y_predictions = xgb_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_predictions)

sns.heatmap(cm,  annot=True)
print(classification_report(y_test, y_predictions))
test_data
test_data.drop(['Cabin'], axis = 1, inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data.dropna(subset = ['Embarked'], inplace=True)
embarked_dummies = pd.get_dummies(test_data['Embarked'])
test_data_ohe = pd.concat([test_data.drop(['Embarked'], axis =1), embarked_dummies], axis =1)
#Importing LabelEncoder from sklearn

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

test_data_ohe['Sex'] = le.fit_transform(test_data_ohe['Sex'])
test_data_ohe.drop(['PassengerId', 'Name', 'Fare', 'Ticket'], axis = 1,inplace=True)
predictions = xgb_classifier.predict(test_data_ohe)
predictions
predictionsDF = pd.DataFrame({

    'PassengerId':test_data['PassengerId'],

    'Survived':predictions

})
predictionsDF
predictionsDF = predictionsDF.set_index('PassengerId')
predictionsDF.to_csv("predictions.csv")