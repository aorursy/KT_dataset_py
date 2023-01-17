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
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import pandas as pd





import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # setting seaborn default for plots





gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



target = train['Survived']
def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Pclass')
train.isnull().sum()
test.isnull().sum()
train_test_data = [train, test]
for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()
test['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 4, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 1, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
train.head()
test.head()
bar_chart('Title')
train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
sex_mapping = {"male": 0, "female": 1}

for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

bar_chart('Sex')
train["Age"].fillna(train.groupby(["Sex", "Title"])["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby(["Sex", "Title"])["Age"].transform("median"), inplace=True)
for dataset in train_test_data:

    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0,

    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 20), 'Age'] = 1,

    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'Age'] = 2,

    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 3,

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 4,

    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 5,

    dataset.loc[ dataset['Age'] > 60, 'Age'] = 6

bar_chart('Age')
train[train.Pclass == 3][train.Sex == 0]['Survived'].value_counts()
train[train.Pclass == 3][train.Sex == 1]['Survived'].value_counts()
mask = (train['Age'] < 1)

mask2 = (test['Age'] < 1)



train.loc[mask, 'PclassSex'] = 0.63

test.loc[mask2, 'PclassSex'] = 0.63
mask = (train['Pclass'] == 1) & (train['Sex'] == 0)

mask2 = (test['Pclass'] == 1) & (test['Sex'] == 0)

mask3 = (train['Pclass'] == 1) & (train['Sex'] == 1)

mask4 = (test['Pclass'] == 1) & (test['Sex'] == 1)

train['PclassSex'] = 0

test['PclassSex'] = 0

train.loc[mask, 'PclassSex'] = 0.63

test.loc[mask2, 'PclassSex'] = 0.63

train.loc[mask3, 'PclassSex'] = 0.96

test.loc[mask4, 'PclassSex'] = 0.96

mask = (train['Pclass'] == 2) & (train['Sex'] == 0)

mask2 = (test['Pclass'] == 2) & (test['Sex'] == 0)

mask3 = (train['Pclass'] == 2) & (train['Sex'] == 1)

mask4 = (test['Pclass'] == 2) & (test['Sex'] == 1)



train.loc[mask, 'PclassSex'] = 0.15

test.loc[mask2, 'PclassSex'] = 0.15

train.loc[mask3, 'PclassSex'] = 0.92

test.loc[mask4, 'PclassSex'] = 0.92

mask = (train['Pclass'] == 3) & (train['Sex'] == 0)

mask2 = (test['Pclass'] == 3) & (test['Sex'] == 0)

mask3 = (train['Pclass'] == 3) & (train['Sex'] == 1)

mask4 = (test['Pclass'] == 3) & (test['Sex'] == 1)



train.loc[mask, 'PclassSex'] = 0.14

test.loc[mask2, 'PclassSex'] = 0.14

train.loc[mask3, 'PclassSex'] = 0.5

test.loc[mask4, 'PclassSex'] = 0.5
train['PclassSex'].value_counts()
train[train.Pclass == 2]['Cabin'].value_counts()
train[train.Pclass == 3]['Cabin'].value_counts()
test['Cabin'].value_counts()
for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

    

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train[train.Pclass == 2]['Cabin'].value_counts()
for dataset in train_test_data:

    dataset['Ticket'] = dataset['Ticket'].str[:1]

    

train['Ticket'].value_counts()
cabin_mapping = {"3": 0, "2": 0, "1": 0, "4": 0, "7": 0, "6": 0, "5": 0, "8": 0, "9": 0, "P": 1, "S": 2, "C": 3, "A": 4, "W": 0, "F": 0, "L": 0}

for dataset in train_test_data:

    dataset['Ticket'] = dataset['Ticket'].map(cabin_mapping)
train[train.Ticket == 0]['Survived'].value_counts()
train[train.Ticket == 1]['Survived'].value_counts()
train[train.Ticket == 2]['Survived'].value_counts()
train[train.Ticket == 3]['Survived'].value_counts()
train[train.Ticket == 4]['Survived'].value_counts()
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1





train[train.FamilySize > 1][train.Age <= 1]['Survived'].value_counts()
train[train.FamilySize > 1][train.Age > 1]['Survived'].value_counts()
train[train.FamilySize == 1][train.Age <= 1]['Survived'].value_counts()
train[train.FamilySize == 1][train.Age > 1]['Survived'].value_counts()
mask = (train['FamilySize'] >= 2) & (train['Age'] <= 1)

mask2 = (test['FamilySize'] >= 2) & (test['Age'] <= 1)

mask3 = (train['FamilySize'] >= 2) & (train['Age'] > 1)

mask4 = (test['FamilySize'] >= 2) & (test['Age'] > 1)





train['FamilySurvive'] = 0

test['FamilySurvive'] = 0



train.loc[mask, 'FamilySurvive'] = 0.46

test.loc[mask2, 'FamilySurvive'] = 0.46

train.loc[mask3, 'FamilySurvive'] = 0.36

test.loc[mask4, 'FamilySurvive'] = 0.36
train['FamilySurvive'].value_counts()
mask = (train['Ticket'] == 0)

mask2 = (test['Ticket'] == 0)

mask3 = (train['Ticket'] == 1)

mask4 = (test['Ticket'] == 1)

mask5 = (train['Ticket'] == 2)

mask6 = (test['Ticket'] == 2)

mask7 = (train['Ticket'] == 3)

mask8 = (test['Ticket'] == 3)

mask9 = (train['Ticket'] == 4)

mask10 = (test['Ticket'] == 4)



train['TicketSurvive'] = 0

test['TicketSurvive'] = 0



train.loc[mask, 'TicketSurvive'] = 0.61

test.loc[mask2, 'TicketSurvive'] = 0.61

train.loc[mask3, 'TicketSurvive'] = 0.65

test.loc[mask4, 'TicketSurvive'] = 0.65

train.loc[mask5, 'TicketSurvive'] = 0.32

test.loc[mask6, 'TicketSurvive'] = 0.32

train.loc[mask7, 'TicketSurvive'] = 0.34

test.loc[mask8, 'TicketSurvive'] = 0.34

train.loc[mask9, 'TicketSurvive'] = 0.07

test.loc[mask10, 'TicketSurvive'] = 0.07
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, 100))

facet.add_legend()

 

plt.show()
train["Fare"].fillna(0, inplace=True)

test['Fare'].fillna(0, inplace=True)

# for data in train_test_data:

#     data['Cabin'].fillna('U', inplace=True)

for dataset in train_test_data:

    dataset.loc[(dataset['Fare']>=0)&(dataset['Fare']<=10), 'Fare'] = 1,

    dataset.loc[(dataset['Fare']>10)&(dataset['Fare']<=20), 'Fare'] = 2,

    dataset.loc[(dataset['Fare']>20)&(dataset['Fare']<=30), 'Fare'] = 3,

    dataset.loc[(dataset['Fare']>30)&(dataset['Fare']<=40), 'Fare'] = 4,

    dataset.loc[(dataset['Fare']>40), 'Fare'] = 5
train.drop('Ticket', axis=1, inplace=True)

test.drop('Ticket', axis=1, inplace=True)

train.drop('Embarked', axis=1, inplace=True)

test.drop('Embarked', axis=1, inplace=True)

train.drop('SibSp', axis=1, inplace=True)

test.drop('SibSp', axis=1, inplace=True)

train.drop('Parch', axis=1, inplace=True)

test.drop('Parch', axis=1, inplace=True)

train.drop('Sex', axis=1, inplace=True)

test.drop('Sex', axis=1, inplace=True)

train.drop('Pclass', axis=1, inplace=True)

test.drop('Pclass', axis=1, inplace=True)

train = train.drop(['PassengerId'], axis=1)

train = train.drop('Survived', axis=1)

train.head()
test.head()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = RandomForestClassifier(n_estimators=13)

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

round(np.mean(score)*100, 2)
clf = SVC()

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

round(np.mean(score)*100, 2)
clf = GaussianNB()

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

round(np.mean(score)*100, 2)
clf = DecisionTreeClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)

round(np.mean(score)*100, 2)
clf = SVC()

clf.fit(train, target)



test_data = test.drop("PassengerId", axis=1).copy()

prediction = clf.predict(test_data)



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission5.csv', index=False)



submission = pd.read_csv('submission5.csv')

submission.head()