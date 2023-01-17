#importing libraries

#for analysis

import numpy as np

import pandas as pd



# for visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#for machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
#Aquring data for this we will use pandas 



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#Check the dataframe

train.head()
train.shape
test.head()
test.shape
train.columns.values
train.info()

print ('_')*50

test.info()
train.describe()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
age = sns.FacetGrid(train, col="Sex", row="Survived", margin_titles=True)

age.map(plt.hist, "Age");
pclass = sns.FacetGrid(train, col="Sex", row="Survived", margin_titles=True)

pclass.map(plt.hist, "Pclass");
#Fare

train['Fare'].plot(kind='hist')

plt.show()
pclass = sns.FacetGrid(train, col="Survived", margin_titles=True)

pclass.map(plt.hist, "Fare");
train['family'] = train['SibSp'] + train['Parch']

test['family'] = test['SibSp'] + test['Parch']
sns.factorplot('family','Survived',data=train,size=5)

plt.show()
train.drop(['SibSp','Parch'],axis=1,inplace=True)

test.drop(['SibSp','Parch'],axis=1,inplace=True)
#droping unnesscery data

train = train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

test = test.drop(['Name','Ticket','Cabin'],axis=1)
train.shape
train[train.Embarked.isnull()]
sns.barplot('Sex', 'Fare','Embarked', data=train,)

plt.show()
train["Embarked"] = train["Embarked"].fillna('C')
test[test.Fare.isnull()]
test.describe()
test["Fare"] = test["Fare"].fillna('35')
train[train.Age.isnull()]
train["Age"] = train["Age"].fillna('29')

test["Age"] = train["Age"].fillna('29')
train['sex'] = train.Sex.map({'female':0,'male':1})

test['sex'] = test.Sex.map({'female':0,'male':1})

test.head(),train.head()
train['embarked'] = train.Embarked.map({'S':1,'C':2,'Q':3})

test['embarked'] = test.Embarked.map({'S':1,'C':2,'Q':3})

test.head(),train.head()
train.drop(['Sex','Embarked'],axis=1,inplace=True)

test.drop(['Sex','Embarked'],axis=1,inplace=True)
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
#Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_prep = logreg.predict(X_test)

logreg.score(X_train, Y_train) 
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn.score(X_train, Y_train) 
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

random_forest.score(X_train, Y_train)
result = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

result.to_csv('titanic_result.csv', index=False)