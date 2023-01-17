import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import random as rnd
from sklearn.linear_model import LogisticRegression
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
combine=[train,test]
train.describe()
train.columns

train.head()
train.describe(include=['O'])
train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by=['Survived'], ascending=True)
train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by=['Survived'], ascending=True)
train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by=['Survived'], ascending=True)
train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by=['Survived'], ascending=True)
train['family']=train['SibSp']+train['Parch']+1
train[['family','Survived']].groupby(['family'],as_index=False).mean().sort_values(by=['Survived'], ascending=True)
test['family']=test['SibSp']+test['Parch']+1
test.head()
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
grid=sns.FacetGrid(train, col='Survived', row='Pclass', height=2, aspect=1.5)
grid.map(plt.hist,'Age')
grid = sns.FacetGrid(train, row='Embarked', col='Survived', height=2, aspect=1.5)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
train=train.drop(['Ticket','Cabin'], axis=1)
test=test.drop(['Ticket','Cabin'], axis=1)
combine=[train,test]
print(train.shape)
print(test.shape)
for dataset in combine:
    dataset['Title']=dataset.Name.str.extract(' ([A-Za-z]+)\.')
pd.crosstab(train['Title'], train['Sex'])    
for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Capt','Col','Countess','Don','Jonkheer','Lady','Major','Rev','Sir','Dr'],'Rare')
    dataset['Title']=dataset['Title'].replace('Mlle','Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Miss')
    dataset['Title']=dataset['Title'].replace('Ms','Miss')

    
pd.crosstab(train['Title'], train['Sex'])
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()
train=train.drop(['Name','PassengerId'], axis=1)
test=test.drop(['Name'], axis=1)
train["Sex"] = train["Sex"].map({"male": 0, "female":1})
test["Sex"] = test["Sex"].map({"male": 0, "female":1})
    
train['Sex'].head()
train.head()
train['Embarked'].unique()
title_mapping = {"S": 1, "C": 2, "Q": 3, np.NaN: 4}

train['Embarked'] = train['Embarked'].map(title_mapping)
test['Embarked'] = test['Embarked'].map(title_mapping)

train.head()
train.info()
for i in range(0,2):
    for j in range(0,3):
        #print(i,j+1)
        temp_dataset=train[(train['Sex']==i) &  (train['Pclass']==j+1)]['Age'].dropna()
        #print(temp_dataset)
        #print(str(temp_dataset.median())+"  "+str(i)+"  "+str(j+1))
        train.loc[(train.Age.isnull()) & (train.Sex==i) & (train.Pclass==j+1),'Age']=int(temp_dataset.median())
        
for i in range(0,2):
    for j in range(0,3):
        #print(i,j+1)
        temp_dataset=test[(test['Sex']==i) &  (test['Pclass']==j+1)]['Age'].dropna()
        #print(temp_dataset)
        #print(str(temp_dataset.median())+"  "+str(i)+"  "+str(j+1))
        test.loc[(test.Age.isnull()) & (test.Sex==i) & (test.Pclass==j+1),'Age']=int(temp_dataset.median())        
train.head(15)
train['Age'] = pd.cut(train['Age'], bins=[0, 12, 50, 200], labels=['Child','Adult','Elder'])
test['Age'] = pd.cut(test['Age'], bins=[0, 12, 50, 200], labels=['Child','Adult','Elder'])
train['Age'].head()
title_mapping = {'Child': 1, 'Adult': 2, 'Elder': 3}

train['Age'] = train['Age'].map(title_mapping).astype(int)
test['Age'] = test['Age'].map(title_mapping).astype(int)
    

train.head()

train=train.drop(['SibSp','Parch'], axis=1)

test=test.drop(['SibSp','Parch'], axis=1)
train.head()
train.info()
test.info()
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.info()
train['Fare']=(train['Fare']- train['Fare'].mean())/np.std(train['Fare'])
train.head()

test['Fare']=(test['Fare']- test['Fare'].mean())/np.std(test['Fare'])
test.head()
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)

