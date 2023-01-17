import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set
%matplotlib inline

#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


#use k-fold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
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
#Read and check titanic dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()

test.head()
train.info()
print('-'*40)
test.info()
train.isnull().sum()

test.isnull().sum()
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
g= sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train, col= 'Survived', row ='Pclass', size= 2.2, aspect = 1.6)
grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)
grid.add_legend()
grid = sns.FacetGrid(train, row = 'Embarked', size= 2.2, aspect = 1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'deep')
grid.add_legend()
grid = sns.FacetGrid(train, row = 'Embarked', col = 'Survived', size = 2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci=None)
grid.add_legend()
#Feature enginnering
train.describe(include="all")
train = train.drop(['Cabin','Ticket'],axis=1)
test = test.drop(['Cabin','Ticket'],axis=1)
train.head()

southampton = train[train["Embarked"]=="S"].shape[0]
print("S=",southampton)
cherbourg = train[train["Embarked"]=="C"].shape[0]
print("C=",cherbourg)
queenstown = train[train["Embarked"]=="Q"].shape[0]
print("Q=",queenstown)


train = train.fillna({"Embarked":"S"})
train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values('Survived', ascending=False)

combine = [train, test]
embarked_mapping = {"S":1,"C":2,"Q":3}
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)

train.head()

test.head()
#name vlaue setting
combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    
pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona','Countess','Lady','Sir'],'Rare')
#     dataset['Title']=dataset['Title'].replace(['Countess','Lady','Sir'], 'Royal')
    dataset['Title']=dataset['Title'].replace(['Mlle','Ms'],'Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Mrs')
    
    
train[['Title','Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Royal":5,"Rare":6}
for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)
    
train.head()
train = train.drop(['Name','PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]
train.head()
test.head()
sex_mapping = {"male":0,"female":1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)

train.head()
grid = sns.FacetGrid(train, row='Pclass', col = 'Sex', size= 2.2 , aspect=1.6)
grid.map(plt.hist, 'Age',alpha=0.5, bins=20)
grid.add_legend()
guess_ages = np.zeros((2,3))
guess_ages
for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] ==j+1)]['Age'].dropna()
            
            age_guess = guess_df.median()
            guess_ages[i,j] = int (age_guess/0.5 + 0.5) * 0.5
            
            
    for i in range(0,2): 
        for j in range(0,3):
            dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex'] ==i) & (dataset['Pclass'] == j+1), 'Age'] = guess_ages[i,j]
            
    dataset['Age'] = dataset['Age'].astype(int)
    
train.head()
train['AgeBand'] = pd.cut(train['Age'],5)
train[['AgeBand','Survived']].groupby(['AgeBand'], as_index = False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16,'Age'] =0
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=32), 'Age']=1
    dataset.loc[(dataset['Age']>32) & (dataset['Age']<=48), 'Age']=2
    dataset.loc[(dataset['Age']>48) & (dataset['Age']<=64), 'Age']=3
    dataset.loc[dataset['Age']>64, 'Age']=4
    
train.head()
train = train.drop(["AgeBand"], axis=1)
combine = [train,test]
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']+ 1
    
train[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',ascending=False)
for dataset in combine:
    dataset['IsAlone']=0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] =1
    
train[['IsAlone','Survived']].groupby('IsAlone',as_index=False).mean().sort_values(by='Survived', ascending=False)
train = train.drop(['Parch','SibSp','FamilySize'], axis=1)
test = test.drop(['Parch','SibSp','FamilySize'], axis=1)
combine = [train, test]
train.head()
test['Fare'].fillna(test['Fare'].median(), inplace=True)
test.head()
test.info()
train['FareBand'] = pd.qcut(train['Fare'],4)
train[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='Survived', ascending = False)
for dataset in combine:
    dataset.loc[dataset['Fare'] <=7.91, 'Fare']=0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] =1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] =2
    dataset.loc[dataset['Fare'] > 31.0, 'Fare'] =3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train = train.drop(['FareBand'], axis=1)
combine = [train, test]

train.head()
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    
train.loc[:,['Age*Class','Age','Pclass']].head()
train.head()
test.head()
# Data Modeling

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test = test.drop("PassengerId", axis=1)

X_train.shape, Y_train.shape, X_test.shape
k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)
clf = RandomForestClassifier()  
clf.fit(X_train,Y_train)
acc_rdf = round(clf.score(X_train,Y_train)*100, 2)
print(acc_rdf)
prediction = clf.predict(X_test)
print(prediction)
submission = pd.DataFrame({
    "PassengerId" : test['PassengerId'],
    "Survived" : prediction
})

submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission.head()


