import pandas as pd

import numpy as np
d = pd.read_csv("../input/train.csv")
d.head()
from sklearn.tree import DecisionTreeClassifier
d['Male']=(d['Sex']=='male')

n = d['Age'].mean()

d['Class1']=(d['Pclass']==1)

d['Class2']=(d['Pclass']==2)



d['Age'].fillna(n, inplace=True)
X = d.loc[:, ['Class1', 'Class2','Male', 'Age', 'SibSp', 'Parch', 'Fare']]

y = d['Survived']
thisclf = DecisionTreeClassifier()
thisclf.fit(X,y)
d['predicted'] = thisclf.predict(X)
from sklearn.metrics import accuracy_score
accuracy_score(y, d['predicted'])
t = pd.read_csv("../input/test.csv")
t['Male']=(t['Sex']=='male')

nn = t['Age'].mean()

t['Class1']=(t['Pclass']==1)

t['Class2']=(t['Pclass']==2)



t['Age'].fillna(nn, inplace=True)



f = t['Fare'].mean()

t['Fare'].fillna(f, inplace=True)
X_t = t.loc[:, ['Class1', 'Class2','Male', 'Age', 'SibSp', 'Parch', 'Fare']]
t['Survived'] = thisclf.predict(X_t)
t_out = t.loc[:,['PassengerId','Survived']]
t_out.to_csv("out.csv")