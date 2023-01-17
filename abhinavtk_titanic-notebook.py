#data analysis and wrangling

import numpy as np

import pandas as pd

import random as rnd

import matplotlib.pyplot as plt
# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
combine = [train, test]
train.head(10)
train.info()
train.describe()
train.describe(include=['O'])
test.head()
train.plot.scatter(x = "Age", y = 'Pclass', c='Parch')
train.isnull().sum()
test.isnull().sum()
a = train[['Sex','Survived']].groupby('Sex').mean()

a
b = train[['Pclass','Survived']].groupby('Pclass', as_index = False).mean().sort_values('Survived', ascending = False)
#visualization
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', color='r', bins=10)
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')

grid.add_legend()
grid = sns.FacetGrid(train, row ='Embarked', col='Survived' )

grid.map(sns.barplot, 'Sex','Fare', alpha=1, ci=None)

grid.add_legend()
train = train.drop(['Ticket','Cabin'],axis=1)

test = test.drop(['Ticket','Cabin'],axis=1)
combine = [train, test]
train[['Name']]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=True)
train
pd.crosstab(train['Sex'], train['Title'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train.head()
train = train.drop(['Name','PassengerId'],axis=1)

test = test.drop(['Name','PassengerId'],axis=1)
train['Sex']= train['Sex'].map({'female':0, 'male':1})

test['Sex']= test['Sex'].map({'female':0, 'male':1})

combine = [train, test]
age_guess = np.zeros((2,3))
for ds in combine:

    for i in range(0,2):

        for j in range(0,3):

            d1 = ds[(ds['Sex']==i) & (ds['Pclass']==j+1)]['Age'].dropna()

            

            a = d1.median()

            age_guess[i,j] = a

    print(age_guess)

    for i in range(0,2):

        for j in range(0,3):

            ds.loc[(ds.Age.isnull()) & (ds['Sex']==i) & (ds['Pclass']==j+1), 'Age'] = age_guess[i,j]

    ds['Age']= ds['Age'].astype(int)
for data in combine:

    data['AgeBand']= pd.cut(data['Age'],5)

train
train[['AgeBand','Survived']].groupby('AgeBand').mean().sort_values('AgeBand')
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train
train['Family'] = train['SibSp']+ train['Parch'] +1

test['Family'] = test['SibSp']+ test['Parch'] +1

train
train.drop(['SibSp','Parch','AgeBand'], axis=1, inplace=True)
test.drop(['SibSp','Parch','AgeBand'], axis=1, inplace=True)
test
train['Isalone']=0

train.loc[train['Family']==1, 'Isalone'] = 1
test['Isalone']=0

test.loc[test['Family']==1, 'Isalone'] = 1
train.drop('Family',axis=1, inplace=True)
test.drop('Family',axis=1, inplace=True)
train[['Isalone','Survived']].groupby('Isalone').mean()
emb_mode=train.Embarked.dropna().mode()[0]
train.Embarked.fillna(emb_mode, inplace = True)
train['Embarked'] = train['Embarked'].map({'S':0, "C":1, "Q":2}).astype(int)

train
test['Embarked'] = test['Embarked'].map({'S':0, "C":1, "Q":2}).astype(int)
test
test.loc[np.where(test['Fare'].isnull()==True)]
a = test[(test['Pclass']==3) & (test['Sex']==1) & (test['Title']==1)]['Fare'].mode()

type(a)
test.Fare.fillna(a[0], inplace=True)
test.loc[np.where(test['Fare'].isnull()==True)]
#Create fare bands
train['Fare band'] = pd.qcut(train['Fare'], 4)
test['Fare band'] = pd.qcut(test['Fare'], 4)
train[['Fare band','Survived']].groupby('Fare band').mean().sort_values('Survived')
combine = [train, test]
for data in combine:

    data.loc[data['Fare']<= 7.91, 'Fare'] = 0

    data.loc[(data['Fare']> 7.91) & (data['Fare']<= 14.454), 'Fare'] = 1

    data.loc[(data['Fare']> 14.454) & (data['Fare']<= 31.0), 'Fare'] = 2

    data.loc[(data['Fare']> 31.0) & (data['Fare']<= 512.329), 'Fare'] = 3
for data in combine:

    data.drop('Fare band', axis=1, inplace=True)

    data['Fare']= data['Fare'].astype(int)
for data in combine:

    data['Age*Class']=data['Age']*data['Pclass']
train.head(10)
test.head(10)
#Model and prediction
X_train = train.drop('Survived', axis=1)
Y_train = train['Survived']
X_test = test
X_train.shape, Y_train.shape, X_test.shape
# machine learning

# This is a supervised classification problem.

# We can use these algorithms: Logistic Regression, KNN, SVM, Naïve Bayes Classifier, Decision Tree, 

# Random Forest, Neural Networks, ANN, RVM

# I'm using Decision Tree.

from sklearn.tree import DecisionTreeClassifier
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
Y_pred = decision_tree.predict(X_test)
output = pd.DataFrame({'Survived':Y_pred})
output