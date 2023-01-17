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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test_data.info()
train_data.isnull().sum()
test_data.isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
def bar_chart(feature):

    survived = train_data[train_data["Survived"]==1][feature].value_counts()

    dead = train_data[train_data["Survived"]==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index=['Survived','Dead']

    df.plot(kind='bar', stacked=True, figsize=(10,5))
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
train_test_data=[train_data, test_data]

for dataset in train_test_data:

    dataset['Title']=dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train_data['Title'].value_counts()
test_data['Title'].value_counts()
title_map={"Mr": 0, "Miss": 1, "Mrs": 2,

          "Master": 3, "Dr": 3, "Rev": 3, "Mlle":3, "Col": 3, "Major": 3, "Mme": 3,

           "Countess": 3, "Sir": 3, "Don": 3, "Ms": 3, "Jonkheer": 3, "Capt": 3, "Lady": 3, "Dona": 3}

for dataset in train_test_data:

    dataset['Title']=dataset['Title'].map(title_map)
train_data.head()

test_data.head()

bar_chart('Title')
train_data.drop("Name", axis=1, inplace=True)

test_data.drop("Name", axis=1, inplace=True)
train_data.head()
test_data.head()
sex_map={"male": 0, "female": 1}

for dataset in train_test_data:

    dataset['Sex']=dataset['Sex'].map(sex_map)
bar_chart('Sex')
train_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"), inplace=True)

test_data["Age"].fillna(test_data.groupby("Title")["Age"].transform("median"), inplace=True)

test_data.info()
#Binning of data for Age

for dataset in train_test_data:

    dataset.loc[dataset['Age']<=16, 'Age']= 0

    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=26), 'Age']= 1

    dataset.loc[(dataset['Age']>26) & (dataset['Age']<=36), 'Age']= 2

    dataset.loc[(dataset['Age']>36) & (dataset['Age']<=62), 'Age']= 3

    dataset.loc[dataset['Age']>62, 'Age']= 4
train_data.head()

bar_chart('Age')
Pclass1= train_data[train_data['Pclass']==1]['Embarked'].value_counts()

Pclass2= train_data[train_data['Pclass']==2]['Embarked'].value_counts()

Pclass3= train_data[train_data['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index=['1stclass', '2ndclass', '3rdclass']

df.plot(kind='bar', stacked=True, figsize=(10,5))
for dataset in train_test_data:

    dataset["Embarked"]=dataset["Embarked"].fillna('S')
embarked_map={'S': 0, 'C': 1, 'Q': 2}

for dataset in train_test_data:

    dataset["Embarked"]=dataset["Embarked"].map(embarked_map)
train_data.head()
train_data["Fare"].fillna(train_data.groupby("Pclass")["Fare"].transform("median"),inplace=True)

test_data["Fare"].fillna(test_data.groupby("Pclass")["Fare"].transform("median"),inplace=True)
train_data.info()
test_data.info()
for dataset in train_test_data:

    dataset.loc[dataset['Fare']<=17, 'Fare']= 0,

    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30), 'Fare']= 1,

    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100), 'Fare']= 2,

    dataset.loc[dataset['Fare']>100, 'Fare']= 3,
train_data.Cabin.value_counts()
for dataset in train_test_data:

    dataset['Cabin']=dataset['Cabin'].str[:1]
Pclass1= train_data[train_data['Pclass']==1]['Cabin'].value_counts()

Pclass2= train_data[train_data['Pclass']==2]['Cabin'].value_counts()

Pclass3= train_data[train_data['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index=['1stclass', '2ndclass', '3rdclass']

df.plot(kind='bar', stacked=True, figsize=(10,5))
cabin_map={'A':0, 'B':0.4, 'C':0.8, 'D':1.2, 'E':1.6, 'F':2.0, 'G':2.4, 'T':2.8}

for dataset in train_test_data:

    dataset['Cabin']=dataset['Cabin'].map(cabin_map)
train_data.head()
train_data["Cabin"].fillna(train_data.groupby("Pclass")["Cabin"].transform("median"),inplace=True)

test_data["Cabin"].fillna(test_data.groupby("Pclass")["Cabin"].transform("median"),inplace=True)
train_data.head()
train_data["FamilySize"]=train_data["SibSp"]+train_data["Parch"]+1

test_data["FamilySize"]=test_data["SibSp"]+test_data["Parch"]+1
train_data.head()
family_map={1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2.0, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4 }

for dataset in train_test_data:

    dataset["FamilySize"]=dataset["FamilySize"].map(family_map)
train_data.head()
features_drop=["Ticket", "SibSp", "Parch"]

train_data=train_data.drop(features_drop, axis=1)

test_data=test_data.drop(features_drop, axis=1)

train_data=train_data.drop(["PassengerId"], axis=1)
train_data.head()
train=train_data.drop(['Survived'],axis=1)

target=train_data['Survived']



train.shape, target.shape
train.head(10)
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import numpy as np
train_data.info()
#Cross validation

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

K_Fold=KFold(n_splits=10, shuffle=True, random_state=0)
clf=KNeighborsClassifier(n_neighbors=13)

scoring='accuracy'

score=cross_val_score(clf, train, target, cv=K_Fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100,2)
#RandomForestClassifier

clf=RandomForestClassifier(n_estimators=13)

scoring='accuracy'

score=cross_val_score(clf, train, target, cv=K_Fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100,2)
#DecisionTreeClassifier

clf=DecisionTreeClassifier()

scoring='accuracy'

score=cross_val_score(clf, train, target, cv=K_Fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100,2)
#NaiveBayes

clf=GaussianNB()

scoring='accuracy'

score=cross_val_score(clf, train, target, cv=K_Fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100,2)
#SVC

clf=SVC()

scoring='accuracy'

score=cross_val_score(clf, train, target, cv=K_Fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100,2)
test_data.head()
test_data.info()
train_data.info()
clf= SVC()

clf.fit(train,target)

test=test_data.drop('PassengerId', axis=1).copy()

prediction=clf.predict(test)
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": prediction})

submission.to_csv("submission.csv", index=False)
submission = pd.read_csv('submission.csv')

submission.head()