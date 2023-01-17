import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
train = pd.read_csv('../input/titanic/train.csv')

train.head()
y_test = pd.read_csv('../input/titanic/gender_submission.csv')

y_test.head()
test = pd.read_csv('../input/titanic/test.csv')

test.head()
sns.heatmap(train.corr(), annot=True, linewidths=1, cmap='twilight')
sns.countplot(data=train, x='Survived',hue='Sex', palette='cool')
sns.countplot(data=train, x='Survived',hue='Pclass', palette='viridis')
sns.boxplot(data=train, x="Survived", y='Age', palette='cool')
sns.boxplot(data=train, x="Pclass", y='Age', palette='viridis')
sns.countplot(train['SibSp'],palette='twilight')
sns.distplot(train['Fare'],bins=50, color='#4A235A')
sns.heatmap(train.isnull(), cbar=False, yticklabels=False)
sns.heatmap(test.isnull(), cbar=False, yticklabels=False)
train['Age'].fillna(value=train['Age'].mean(), inplace = True)

train.drop('Cabin', axis=1, inplace=True)

train.dropna(inplace=True)

train.isnull().sum()
test['Age'].fillna(value=test['Age'].mean(), inplace = True)

test['Fare'].fillna(value=test['Fare'].mean(), inplace = True)

test.drop('Cabin', axis=1, inplace=True)

train.isnull().sum()
pclass = pd.get_dummies(train['Pclass'], drop_first=True)

sex = pd.get_dummies(train['Sex'], drop_first=True)

embarked = pd.get_dummies(train['Embarked'], drop_first=True)

train = pd.concat([train, pclass, sex, embarked], axis=1)

train.head()
pclass2 = pd.get_dummies(test['Pclass'], drop_first=True)

sex2 = pd.get_dummies(test['Sex'], drop_first=True)

embarked2 = pd.get_dummies(test['Embarked'], drop_first=True)

test = pd.concat([test, pclass2, sex2, embarked2], axis=1)

test.head()
train = train.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Ticket','Embarked'], axis=True)

train.head()
test = test.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Ticket','Embarked'], axis=True)

test.head()
x_train = train.drop('Survived', axis=1)

y_train = train['Survived']

y_test = y_test['Survived']
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(max_iter=500)

logr.fit(x_train, y_train)

pred = logr.predict(test)
from sklearn.metrics import confusion_matrix
plt.axes().set_title("Confusion Matrix")

cm = pd.DataFrame(confusion_matrix(y_test, pred),columns=["Predicted Negative", "Predicted Positive"], index=["Actual Negative", "Actual Positive"])

sns.heatmap(cm,annot=True, cmap='viridis', fmt="d")
from sklearn.metrics import  precision_score, recall_score, accuracy_score

precision_score(y_test, pred)
recall_score(y_test, pred)
accuracy_score(y_test, pred)
sub_file = pd.read_csv('../input/titanic/gender_submission.csv')

sub_file['Survived'] = pred

sub_file.to_csv("submission.csv", index=False)