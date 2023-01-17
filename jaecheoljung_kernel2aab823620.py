# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
train.describe()
train.describe(include=['O'])
for dataset in [train, test]:
    dataset['Age'] = dataset['Age'].fillna(dataset.Age.dropna().mode()[0])
    dataset['Cabin'] = dataset['Cabin'].fillna('NONE')
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset.Embarked.dropna().mode()[0])
    dataset['Fare'] = dataset['Fare'].fillna(dataset.Fare.dropna().mode()[0])
    
print(train.isnull().sum())
print('-'*10)
print(test.isnull().sum())
for dataset in [train, test]:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['CabinGroup'] = dataset.Cabin.str.slice(0,1)
    dataset['AgeGroup'] = pd.cut(dataset['Age'], 5)
    dataset['FareGroup'] = pd.cut(dataset['Fare'], 2)
    
pd.crosstab(train['Title'], train['Sex'])

pd.crosstab(train['CabinGroup'], train['Sex'])
_ = test['PassengerId']

train = train.drop(['Fare', 'Age', 'SibSp', 'Parch', 'Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)
test = test.drop(['Fare', 'Age', 'SibSp', 'Parch', 'Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1)

le = LabelEncoder()
X = train.drop(['Survived'], axis=1)
Y = train['Survived']

for x in [X, test]:
    x['FareGroup'] = le.fit_transform(x['FareGroup'])
    x['AgeGroup'] = le.fit_transform(x['AgeGroup'])
    x['CabinGroup'] = le.fit_transform(x['CabinGroup'])
    x['Embarked'] = le.fit_transform(x['Embarked'])
    x['Sex'] = le.fit_transform(x['Sex'])
    x['Title'] = le.fit_transform(x['Title'])
    
X.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#model = GradientBoostingClassifier(n_estimators=700, max_depth=7, learning_rate=0.05)
model = RandomForestClassifier(n_estimators=100)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.3, random_state = 2)

model.fit(X_train, Y_train)
predictions = model.predict(X_val)
print('Accuracy: ', accuracy_score(predictions, Y_val))
model.fit(X, Y)
predictions = model.predict(X)
print('Accuracy: ', accuracy_score(predictions, Y))
y = model.predict(test)
output = pd.DataFrame({'PassengerId': _, 'Survived': y})
output.to_csv('submission.csv', index=False)

output.head()