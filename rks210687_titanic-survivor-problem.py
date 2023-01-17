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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv("../input/train.csv");
train.head()
train['PassengerId'].head(20)
train['PassengerId'].tail(20)
train['PassengerId'].describe()
train['Survived'].corr(train['PassengerId'])
train.drop(['PassengerId'], axis=1, inplace=True)
train.head()
train['Pclass'].head(20)
train['Pclass'].tail(20)
train['Pclass'].describe()
train['Pclass'].unique()
train['Survived'].corr(train['Pclass'])
import seaborn as sns
sns.regplot(x="Pclass", y="Survived", data=train);
train.groupby(['Pclass'])['Survived'].mean()
train['Pclass'] = train['Pclass'].astype('category')
train['Pclass'].dtype
train['Name'].head(20)
train['Title'] = train['Name'].str.replace('(.*, )|(\\..*)', '')
train.head()
train.groupby(['Title'])['Survived'].mean()
train.groupby(['Title'])['Survived'].count()
TITLE = {
    'Mr': 1, 
    'Mrs': 2, 
    'Miss': 3, 
    'Master': 4, 
    'Don': 5, 
    'Rev': 6, 
    'Dr': 7, 
    'Mme': 8, 
    'Ms': 9,
    'Major': 10, 
    'Lady': 11, 
    'Sir': 12, 
    'Mlle': 13, 
    'Col': 14, 
    'Capt': 15, 
    'the Countess': 16,
    'Jonkheer': 17,
    'Dona': 18
}
train['Title'] = train['Title'].replace(TITLE).astype('category')
train.head()
train['Sex'].unique()
train.groupby(['Sex'])['Survived'].mean()
SEX = {
    'male': 1, 
    'female': 2
}
train['Sex'] = train['Sex'].replace(SEX).astype('category')
train.head()
train['Age'].mean()
train['Age'].fillna(29.69911764705882, inplace=True)
train.head(10)
def f(train):
    return int(round(train['Age'] / 5)) + 1;


train['Age1'] = train.apply(f, axis=1).astype('category')
train.head()
train.groupby(['Age1'])['Survived'].mean()
train['family_size'] = train['SibSp'] + train['Parch']
train.groupby(['family_size'])['Survived'].mean()
def fare(train):
    return int(round(train['Fare'] / 5));


train['fare1'] = train.apply(fare, axis=1).astype('category')
train.head()
train.groupby(['fare1'])['Survived'].mean()
train.groupby(['Embarked'])['Survived'].mean()
train['Embarked'].fillna('N', inplace=True)
train.head(10)
train.groupby(['Embarked'])['Survived'].mean()
Embarked = {
    'S': 1, 
    'C': 2,
    'Q': 3,
    'N': 4
}
train['Embarked'] = train['Embarked'].replace(Embarked).astype('category')
train.head()
from sklearn.model_selection import train_test_split
train_X = train[['Pclass', 'Title', 'Sex', 'Age1', 'family_size', 'fare1', 'Embarked']]
train_Y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.4, random_state=4)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

svc = SVC()
svc.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)

linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)

sgd = SGDClassifier()
sgd.fit(X_train, y_train)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
from sklearn import metrics

y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

y_pred = svc.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

y_pred = gaussian.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

y_pred = perceptron.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

y_pred = linear_svc.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

y_pred = sgd.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

y_pred = decision_tree.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

y_pred = random_forest.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
test = pd.read_csv("../input/test.csv");
test.head()
test['Title'] = test['Name'].str.replace('(.*, )|(\\..*)', '')
test['Title'] = test['Title'].replace(TITLE).astype('category')

test['Sex'] = test['Sex'].replace(SEX).astype('category')

test['Age'].fillna(29.69911764705882, inplace=True)
test['Age1'] = test.apply(f, axis=1).astype('category')


test['family_size'] = test['SibSp'] +test['Parch']
test['Fare'].fillna(0, inplace=True)
test['fare1'] = test.apply(fare, axis=1).astype('category')

test['Embarked'].fillna('N', inplace=True)

test['Embarked'] = test['Embarked'].replace(Embarked).astype('category')

test.head()
y_pred = logreg.predict(test[['Pclass', 'Title', 'Sex', 'Age1', 'family_size', 'fare1', 'Embarked']])
y_pred