import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
gender_submission
test
train.head()
train.describe()
train.hist(figsize = (12,12))

plt.show()
train_corr = train.corr()

plt.figure(figsize = (15,15))

sns.heatmap(train_corr,annot = True)
sns.countplot('Sex',hue = 'Survived',data = train)
train.isnull().sum()
test.isnull().sum()
train['Sex'] = train['Sex'].apply(lambda x : 1 if x == 'male' else 0)

train = train.drop(['PassengerId', 'Name', 'Cabin', 'Embarked','Ticket'],axis = 1)

train['Age'] = train['Age'].fillna(train['Age'].mean())



test['Sex'] = test['Sex'].apply(lambda x : 1 if x == 'male' else 0)

test = test.drop(['PassengerId', 'Name', 'Cabin', 'Embarked','Ticket'],axis = 1)

test['Age'] = test['Age'].fillna(test['Age'].mean())

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

x = train.drop(['Survived'],axis=1)

y = train['Survived']

clf = LogisticRegression()

clf.fit(x,y)
clf.score(x,y)
test_predict = clf.predict(test)
test = pd.read_csv("../input/titanic/test.csv")

result = pd.DataFrame({'PassengerId':test['PassengerId'],

                       'Survived':np.array(test_predict)

                      })

result
result.to_csv('result.csv',index=False)