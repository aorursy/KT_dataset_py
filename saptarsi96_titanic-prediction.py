import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/train.csv')
df.head()
columns_target = ['Survived']

columns_train  = ['Pclass','Sex','Age','Fare']
X = df[columns_train]

y = df[columns_target]
X['Sex'].isnull().sum()
X['Pclass'].isnull().sum()
X['Age'].isnull().sum()
X['Fare'].isnull().sum()
X['Age'] = X['Age'].fillna(X['Age'].median())

X['Age'].isnull().sum()
X.head()
d = {'male': 0 ,'female':1}

X['Sex'] = X['Sex'].apply(lambda x:d[x])

X['Sex'].head()
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train,Y_train)
clf.predict(X_test)

X_test.head()
clf.score(X_test,Y_test)
df2 = pd.read_csv('../input/test.csv')
df2.describe()
columns_target = ['Survived']

columns_train  = ['Pclass','Sex','Age','Fare']
X = df2[columns_train]

#y = df2[columns_target]
X['Age'] = X['Age'].fillna(X['Age'].median())

X['Age'].isnull().sum()

d = {'male': 0 ,'female':1}

X['Sex'] = X['Sex'].apply(lambda x:d[x])

X['Sex'].head()
X.describe()
X['Sex'].isnull().sum()

X['Pclass'].isnull().sum()
X['Age'].isnull().sum()
X['Fare'].isnull().sum()
X['Fare'] = X['Fare'].fillna(X['Fare'].mean())

X['Fare'].isnull().sum()
test_passengerIds = df2['PassengerId'].copy()
y_pred = clf.predict(X)
pd.DataFrame({'PassengerId': test_passengerIds, 'Survived': y_pred}).to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head()
submission.describe()