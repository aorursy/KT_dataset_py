import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



X = pd.read_csv("../input/train.csv")



T = pd.read_csv("../input/test.csv")



ids = T['PassengerId']
X.head()
X.tail()
titles = X.columns.values

titles
X.describe()
X.describe(include=['O'])
X[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(X, col='Survived')

g.map(plt.hist, 'Age', bins=100)
X[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
X[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
X[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train = ['Pclass','Sex','Age','SibSp','Parch']

X_train = X[train]

T = T[train]

Y_train = X['Survived']
X_train['Sex'] = pd.get_dummies(X['Sex'])

T['Sex'] = pd.get_dummies(T['Sex'])
X_train.head(7)
X_train['Age'].mean()
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())

T['Age'] = T['Age'].fillna(T['Age'].mean())



X_train.head(10)
T.head(10)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
from sklearn.cross_validation import cross_val_score
score = cross_val_score(knn, X_train, Y_train, cv = 10, scoring = 'accuracy')

print(score)
print(score.mean())
k_range = range(1,31)

k_scores = []



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors = k)

    k_scores.append(cross_val_score(knn, X_train, Y_train, cv = 10, scoring = 'accuracy').mean())
plt.bar(k_range,k_scores, width = 0.2)
logreg = LogisticRegression()
scorel = cross_val_score(logreg, X_train, Y_train, cv = 10, scoring= "accuracy").mean()

scorel
logreg.fit(X_train, Y_train)



Y_predicted = logreg.predict(T)
results = pd.DataFrame({ 'PassengerId' : ids, 'Survived': Y_predicted })
results.head()
results.to_csv( 'titanic_pred.csv' , index = False )
Testing = pd.read_csv('titanic_pred.csv')

Testing.head()