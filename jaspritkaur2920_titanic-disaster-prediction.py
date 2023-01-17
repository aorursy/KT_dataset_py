# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() #setting seaborn default for plots

import os
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.shape
test.shape
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived, dead])

    df.index = ['Survived', 'Dead']

    df.plot(kind = 'bar', stacked = True, figsize = (10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
#names

train_test_data= [train, test] #combining training and test data

for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand = False)
train['Title'].value_counts()
test['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
train.head()
test.head()
bar_chart('Title')
#deleting unnessery feature from dataset

train.drop('Name', axis=1, inplace = True)

test.drop('Name', axis=1, inplace = True)
train.head()
test.head()
# sex

# male=0, female=1

sex_mapping = {"male":0, "female":1}

for dataset in train_test_data:

    dataset["Sex"] = dataset["Sex"].map(sex_mapping)
train.head()
test.head()
bar_chart('Sex')
# age

# some age is missing, we'll use title's median age for missing age

# fill missing age with median age for each title (Mr, Miss, Mrs, Others)

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
train.head(30)

train.groupby("Title")["Age"].transform("median")
facet = sns.FacetGrid(train, hue="Survived", aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()



plt.show()
facet = sns.FacetGrid(train, hue="Survived", aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(0,20)
facet = sns.FacetGrid(train, hue="Survived", aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(30,40)
facet = sns.FacetGrid(train, hue="Survived", aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(40,60)
facet = sns.FacetGrid(train, hue="Survived", aspect=4)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(60)
train.info()
test.info()
for dataset in train_test_data:

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,

    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,

    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,

    dataset.loc[dataset['Age'] > 62, 'Age'] = 4
train.head()
bar_chart('Age')
#Embarked 

#filling missing values

Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class', '2nd index', '3rd class']

df.plot(kind='bar',stacked = True, figsize=(10,5))
#filling out missing data with S embark

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train.head()
#embark mapping S=0, C=1, Q=2

embarked_mapping = {"S":0, "C":1, "Q":2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
# fare 

# fill missing fare with median fare for each Pclass

train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
train.head()
facet = sns.FacetGrid(train, hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Fare', shade = True)

facet.set(xlim=(0,train['Fare'].max()))

facet.add_legend()



plt.show()
facet = sns.FacetGrid(train, hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Fare', shade = True)

facet.set(xlim=(0,train['Fare'].max()))

facet.add_legend()

plt.xlim(0,20)
facet = sns.FacetGrid(train, hue="Survived", aspect=4)

facet.map(sns.kdeplot, 'Fare', shade = True)

facet.set(xlim=(0,train['Fare'].max()))

facet.add_legend()

plt.xlim(0,30)
for dataset in train_test_data:

    dataset.loc[dataset['Fare'] <= 17, 'Fare']=0,

    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare']=1,

    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare']=2,

    dataset.loc[dataset['Fare'] >100, 'Fare']=3
train.head()
# cabin

train.Cabin.value_counts()
for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()

Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()

Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
# fill missing Fare with median fare for each Pclass

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
# familysize

train['Familysize'] = train["SibSp"] + train["Parch"] + 1

test['Familysize'] = test["SibSp"] + test["Parch"] + 1
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Familysize',shade= True)

facet.set(xlim=(0, train['Familysize'].max()))

facet.add_legend()

plt.xlim(0)
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for dataset in train_test_data:

    dataset['Familysize'] = dataset['Familysize'].map(family_mapping)
train.head()
test.head()
features_drop = ['Ticket', 'SibSp', 'Parch']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId'], axis=1)
X = train.drop('Survived', axis=1)

y = train['Survived']



X.shape, y.shape
X.head()
train.info()
# important libraries for predictions

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
# cross-validation

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
# kNN

knn = KNeighborsClassifier(n_neighbors=13)

scoring = 'accuracy'

score = cross_val_score(knn, X, y, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
# random forest

rf = RandomForestClassifier(n_estimators=13)

scoring = 'accuracy'

score = cross_val_score(rf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
# Naive Bayes

nb = GaussianNB()

scoring = 'accuracy'

score = cross_val_score(nb, X, y, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
# SVM

svm = SVC()

scoring = 'accuracy'

score = cross_val_score(svm, X, y, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
#testing

clf = SVC()

clf.fit(X, y)



test_data = test.drop("PassengerId", axis=1).copy()

prediction = clf.predict(test_data)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head()