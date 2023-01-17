import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(5)
train.isnull().sum()
sb.countplot("Survived", data=train)

plt.show()
train['Survived'].mean()

train.groupby(['Sex','Pclass']).mean()

def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True)
bar_chart('Pclass')
bar_chart('Sex')
bar_chart('Embarked')
bar_chart('SibSp')
survived = train[train['Survived']==1][train['Cabin'].isnull()==False]['Survived'].value_counts()

dead = train[train['Survived']==0][train['Cabin'].isnull()==False]['Survived'].value_counts()

df = pd.DataFrame([survived,dead])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True)
survived = train[train['Survived']==1][train['Cabin'].isnull()==True]['Survived'].value_counts()

dead = train[train['Survived']==0][train['Cabin'].isnull()==True]['Survived'].value_counts()

df = pd.DataFrame([survived,dead])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True)
import re

combine=[train,test]

# train_test_df = train.append(test, ignore_index=True)

pattern = re.compile('([A-Za-z]+)\.')

for dataset in combine:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, 

                 "Master": 4, "Dr": 5, "Rev": 5, "Col": 5, "Major": 5, "Mlle": 2,"Countess": 4,

                 "Ms": 2, "Lady": 4, "Jonkheer": 5, "Don": 5, "Dona" : 5, "Mme": 3,"Capt": 5,"Sir": 4 }



for dataset in combine:

    #Mr. is 1

    dataset['Title'] = dataset['Title'].replace(['Capt.', 'Col.', 

        'Don.', 'Dr.', 'Major.', 'Rev.', 'Jonkheer.', 'Dona.'], 'Other.')    #Other. is 5  



    dataset['Title'] = dataset['Title'].replace(['Ms.', 'Mlle.'], 'Miss.')   #Miss. is 2



    dataset['Title'] = dataset['Title'].replace('Mme.', 'Mrs.') # Mrs. is 4



    dataset['Title'] = dataset['Title'].replace(['Lady.', 'Master.', 'Countess.', 'Sir.'], 'Royal.') # Mrs. is 4





pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
train.sample(10)
sex_mapping = {"male":0 , "female":1}

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
# I assume that these 2 features does not affect the result

train.drop('Ticket', axis=1, inplace=True)

test.drop('Ticket', axis=1, inplace=True)
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))
train.drop('Cabin', axis=1, inplace=True)

test.drop('Cabin', axis=1, inplace=True)
train.sample(10)
train = train.fillna({"Embarked": "S"})

test = test.fillna({"Embarked": "S"})
embark_mapping = {"S": 1, "C": 2, "Q":3}

train['Embarked'] = train['Embarked'].map(embark_mapping)

test['Embarked'] = test['Embarked'].map(embark_mapping)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
train.loc[ train['Age'] <= 5, 'Age'] = 0, #Baby

train.loc[(train['Age'] > 5) & (train['Age'] <= 12), 'Age'] = 1, #Child

train.loc[(train['Age'] > 12) & (train['Age'] <= 18), 'Age'] = 2, #Teenager

train.loc[(train['Age'] > 18) & (train['Age'] <= 24), 'Age'] = 3, #Student

train.loc[(train['Age'] > 24) & (train['Age'] <= 35), 'Age'] = 4, #Young Adult

train.loc[(train['Age'] > 35) & (train['Age'] <= 60), 'Age'] = 5, #Adult

train.loc[ train['Age'] > 60, 'Age'] = 6 #Senior
test.loc[ test['Age'] <= 5, 'Age'] = 0, #Baby

test.loc[(test['Age'] > 5) & (test['Age'] <= 12), 'Age'] = 1, #Child

test.loc[(test['Age'] > 12) & (test['Age'] <= 18), 'Age'] = 2, #Teenager

test.loc[(test['Age'] > 18) & (test['Age'] <= 24), 'Age'] = 3, #Student

test.loc[(test['Age'] > 24) & (test['Age'] <= 35), 'Age'] = 4, #Young Adult

test.loc[(test['Age'] > 35) & (test['Age'] <= 60), 'Age'] = 5, #Adult

test.loc[ test['Age'] > 60, 'Age'] = 6 #Senior
train.isnull().sum()
# I assume that Fare attribute won't affect much to survivor rate so I will drop it.

test.drop(['Fare'], axis=1, inplace=True)

train.drop(['Fare'], axis=1, inplace=True)
train.sample(10)
combine
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

import numpy as np
train_data = train.drop(['Survived', 'PassengerId'], axis=1)

target = train['Survived']
train_data.sample(15)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = SVC()

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
clf = RandomForestClassifier(n_estimators=13)

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
clf = LogisticRegression()

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100, 2)
clf = SVC()

clf.fit(train_data, target)



test_data = test.drop("PassengerId", axis=1).copy()

prediction = clf.predict(test_data)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)