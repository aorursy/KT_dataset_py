# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()
#train_data.info()

train_data.isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



def bar_chart(feature):

    survived = train_data[train_data['Survived']==1][feature].value_counts()

    dead = train_data[train_data['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Pclass')
bar_chart('Sex')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
bar_chart('Cabin')
train_test_data = [train_data, test_data] # combining train and test dataset



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_data['Title'].value_counts()
test_data['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
train_data.head()
test_data.head()
bar_chart('Title')
# delete unnecessary feature from dataset

train_data.drop('Name', axis=1, inplace=True)

test_data.drop('Name', axis=1, inplace=True)
sex_mapping = {"male": 0, "female": 1}

for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
# fill missing age with median age for each title (Mr, Mrs, Miss, Others)

train_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"), inplace=True)

test_data["Age"].fillna(test_data.groupby("Title")["Age"].transform("median"), inplace=True)
train_data.groupby("Title")["Age"].transform("median")
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train_data['Age'].max()))

facet.add_legend()

 

plt.show()
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train_data['Age'].max()))

facet.add_legend()

plt.xlim(0, 20)
for dataset in train_test_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,

    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,

    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,

    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
bar_chart('Age')
Pclass1 = train_data[train_data['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train_data[train_data['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train_data[train_data['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train_data.head()
embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
# fill missing Fare with median fare for each Pclass

train_data["Fare"].fillna(train_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test_data["Fare"].fillna(test_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)

train_data.head()
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train_data['Fare'].max()))

facet.add_legend()

 

plt.show()
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train_data['Fare'].max()))

facet.add_legend()

plt.xlim(0, 20)
for dataset in train_test_data:

    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,

    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,

    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,

    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = train_data[train_data['Pclass']==1]['Cabin'].value_counts()

Pclass2 = train_data[train_data['Pclass']==2]['Cabin'].value_counts()

Pclass3 = train_data[train_data['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
# fill missing Fare with median fare for each Pclass

train_data["Cabin"].fillna(train_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test_data["Cabin"].fillna(test_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1

test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'FamilySize',shade= True)

facet.set(xlim=(0, train_data['FamilySize'].max()))

facet.add_legend()

plt.xlim(0)
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for dataset in train_test_data:

    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
features_drop = ['Ticket', 'SibSp', 'Parch']

train_data = train_data.drop(features_drop, axis=1)

test_data = test_data.drop(features_drop, axis=1)

train_data = train_data.drop(['PassengerId'], axis=1)
trainx = train_data.drop('Survived', axis=1)

target = train_data['Survived']



trainx.shape, target.shape
train_data.head()
# Importing Classifier Modules

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



import numpy as np
train_data.info()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
#for all models

scoring = 'accuracy'
clf = KNeighborsClassifier(n_neighbors = 13)

score = cross_val_score(clf, trainx, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# kNN Score

round(np.mean(score)*100, 2)
clf = DecisionTreeClassifier()

score = cross_val_score(clf, trainx, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# decision tree Score

round(np.mean(score)*100, 2)
clf = RandomForestClassifier(n_estimators=13)

score = cross_val_score(clf, trainx, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# Random Forest Score

round(np.mean(score)*100, 2)
clf = GaussianNB()

score = cross_val_score(clf, trainx, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# Naive Bayes Score

round(np.mean(score)*100, 2)
clf = SVC()

score = cross_val_score(clf, trainx, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100,2)
clf = SVC()

clf.fit(trainx, target)



testx = test_data.drop("PassengerId", axis=1).copy()

prediction = clf.predict(testx)
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head()