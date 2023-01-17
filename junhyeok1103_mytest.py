import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train.info()
train.isnull().sum()
test.info()
test.isnull().sum()
msno.matrix(df=train.iloc[:, :], figsize=(6, 4), color=(0.8, 0.5, 0.2))
def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df= pd.DataFrame([survived, dead])

    df.index = ['Survived', 'Dead']

    df.plot(kind = 'bar', stacked='True', figsize=(10,5))
f, ax = plt.subplots(1, 2, figsize=(8, 5))



train[train['Survived']==1]['Sex'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Sex - Survived')

train[train['Survived']==0]['Sex'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[1], shadow=True)

ax[1].set_title('Sex - Dead')

plt.show()
bar_chart('Pclass')
bar_chart("SibSp")
bar_chart("Parch")
bar_chart("Embarked")
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

plot = sns.distplot(train['Fare'], color='b', ax =ax[0])

plot2 = sns.distplot(train['Fare'][train["Survived"]==1], color='b',  ax =ax[1])

plot2 = sns.distplot(train['Fare'][train["Survived"]==0], color='r',  ax =ax[1])

plot= plot.legend(loc='best')
train.describe(include='all')
train = train.drop(['Ticket', 'Cabin'], axis=1)

test = test.drop(['Ticket', 'Cabin'], axis=1)
southhampton= train[train["Embarked"]=='S'].shape[0]

print('S : ', southhampton)

cherbourg= train[train["Embarked"]=='C'].shape[0]

print('C : ',cherbourg)

queenstown = train[train["Embarked"]=='Q'].shape[0]

print('D : ',cherbourg)
train=train.fillna({"Embarked" : "S"})
embarked_mapping = {'S': 1 , "C" : 2 , "Q" : 3}

train["Embarked"] = train["Embarked"].map(embarked_mapping)

test["Embarked"] = test["Embarked"].map(embarked_mapping)

train.head()
combine = [train, test]

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand = False)

pd.crosstab(train['Title'],train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Other')

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mile', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index = False).mean()
train.loc[(train.Age.isnull())&(train.Title=='Mr'),'Age'] = 33

train.loc[(train.Age.isnull())&(train.Title=='Mrs'),'Age'] = 36

train.loc[(train.Age.isnull())&(train.Title=='Master'),'Age'] = 5

train.loc[(train.Age.isnull())&(train.Title=='Miss'),'Age'] = 22

train.loc[(train.Age.isnull())&(train.Title=='Other'),'Age'] = 46



test.loc[(test.Age.isnull())&(test.Title=='Mr'),'Age'] = 33

test.loc[(test.Age.isnull())&(test.Title=='Mrs'),'Age'] = 36

test.loc[(test.Age.isnull())&(test.Title=='Master'),'Age'] = 5

test.loc[(test.Age.isnull())&(test.Title=='Miss'),'Age'] = 22

test.loc[(test.Age.isnull())&(test.Title=='Other'),'Age'] = 46
title_mapping = {'Mr' : 1, 'Miss' : 2, "Mrs" : 3, "Master" : 4, "Royal" : 5, "Rare" : 6 }

for dataset in combine:

    dataset['Title']= dataset["Title"].map(title_mapping)

    dataset['Title']= dataset['Title'].fillna(0)

train.head()
train = train.drop(['Name', 'PassengerId'] , axis = 1)

test = test.drop(['Name', 'PassengerId'] , axis = 1)

combine = [train, test]

train.head()
sex_mapping = {"male" : 0 , "female" : 1}

for dataset in combine:

    dataset['Sex']= dataset['Sex'].map(sex_mapping)
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

train.head()
age_mapping={'Baby' :1, 'Child':2, 'Teenager':3, 'Student':4, 'Young Adult':5, 'Adult':6, 'Senior':7}

train["AgeGroup"]= train["AgeGroup"].map(age_mapping)

test["AgeGroup"]= test["AgeGroup"].map(age_mapping)

train = train.drop(['Age'] , axis =1)

test = test.drop(['Age'], axis = 1)
test.loc[test.Fare.isnull(), 'Fare'] = test['Fare'].mean() 



train['Fare'] = train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

test['Fare'] = test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

fare_plot = sns.distplot(train['Fare'], color='b', ax=ax)

fare_plot = fare_plot.legend(loc='best')
train_data = train.drop(["Survived"], axis=1)

target = train["Survived"]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits = 10 , shuffle =True, random_state = 0)
clf = RandomForestClassifier(n_estimators = 13)
scoring = 'accuracy'

score = cross_val_score(clf, train_data , target, cv= k_fold, n_jobs=1, scoring= scoring)

print(score)
round(np.mean(score)*100,2)
clf = KNeighborsClassifier()
scoring = 'accuracy'

score = cross_val_score(clf, train_data , target, cv= k_fold, n_jobs=1, scoring= scoring)

print(score)
round(np.mean(score)*100,2)
clf =DecisionTreeClassifier()
scoring = 'accuracy'

score = cross_val_score(clf, train_data , target, cv= k_fold, n_jobs=1, scoring= scoring)

print(score)
round(np.mean(score)*100,2)
clf =GaussianNB()
scoring = 'accuracy'

score = cross_val_score(clf, train_data , target, cv= k_fold, n_jobs=1, scoring= scoring)

print(score)
round(np.mean(score)*100,2)
clf =SVC()
scoring = 'accuracy'

score = cross_val_score(clf, train_data , target, cv= k_fold, n_jobs=1, scoring= scoring)

print(score)
round(np.mean(score)*100,2)
clf = SVC()

clf.fit(train_data, target)
prediction = clf.predict(test)
test_Passenger = pd.read_csv('../input/test.csv')

submission = pd.DataFrame({"PassengerId":test_Passenger["PassengerId"], "Survived" : prediction})

submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head()