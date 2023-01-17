import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
print('# of passengers in train dataset: ' + str(len(train)))
sns.countplot(x = 'Survived', data = train)
sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
train['Age'].plot.hist()
train['Fare'].plot.hist()
train.info()
sns.countplot(x = 'SibSp', data = train)
train.isnull().sum()
sns.heatmap(train.isnull())
sns.boxplot(x = 'Pclass', y = 'Age', data = train)
sex = pd.get_dummies(train['Sex'], drop_first = True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

pcl = pd.get_dummies(train['Pclass'],drop_first=True)
train = pd.concat([train,sex,embark,pcl],axis=1)

train.head()
train.drop(['Pclass','Sex','Embarked','Cabin','PassengerId','Name','Ticket'],axis=1, inplace=True)

train.head()
train.isnull().sum()
train_values = {'Age': round(np.mean(train['Age']))}

train = train.fillna(value = train_values)

train.head()
sex = pd.get_dummies(test['Sex'], drop_first = True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)

pcl = pd.get_dummies(test['Pclass'],drop_first=True)
test = pd.concat([test,sex,embark,pcl],axis=1)

test.head()
test.drop(['Pclass','Sex','Embarked','Cabin','PassengerId','Name','Ticket'],axis=1, inplace=True)
test.isnull().sum()
test_values = {'Age':round(np.mean(test['Age'])), 'Fare':round(np.mean(test['Fare']))}

test = test.fillna(value = test_values)

test.head()
X = train.drop('Survived',axis=1)

y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train, y_train)
predections = logmodel.predict(X_test)
print(classification_report(y_test, predections))
print(confusion_matrix(y_test, predections))
print(accuracy_score(y_test, predections))
test_predictions = logmodel.predict(test)
sub_file = pd.read_csv('../input/titanic/gender_submission.csv')

sub_file['Survived'] = test_predictions

sub_file.to_csv('submission.csv',index=False)