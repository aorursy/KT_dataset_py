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
#importing data wraggling libraries

import numpy as np 

import pandas as pd

import random as rd



#importing data visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



#importing machine learning libraries

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")

comb = [df_train, df_test]
print(df_train.columns.values)
df_train.info()
df_train.head()
plt.figure(figsize=(10,8))

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df_train.info()

print('*'*40)

df_test.info()
df_train.describe()
df_train.Sex.value_counts()
df_train.Pclass.value_counts()
df_train.describe(include=[object])
df_train.describe(include=[object])
sns.set_style('whitegrid')

sns.countplot(x='Survived', data=df_train)
sns.set_style('whitegrid')

sns.countplot(x='Survived', hue='Pclass', data=df_train)
df_train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean()
sns.set_style('whitegrid')

sns.countplot(x='Survived', hue='Sex', data=df_train)
df_train[['Sex','Survived']].groupby('Sex', as_index=False).mean()
sns.set_style('whitegrid')

sns.countplot(x='Survived', hue='SibSp', data=df_train)
df_train[['SibSp','Survived']].groupby('SibSp', as_index=False).mean().sort_values('Survived', ascending=False)
sns.set_style('whitegrid')

sns.countplot(x='Survived', hue='Parch', data=df_train)
df_train[['Parch','Survived']].groupby('Parch', as_index=False).mean().sort_values('Survived', ascending=False)
a = sns.FacetGrid(df_train, col='Survived')

a.map(plt.hist,'Age',bins=30)
a = sns.FacetGrid(df_train, col='Survived', row='Pclass')

a.map(plt.hist, 'Age', alpha=.5, bins=20)

a.add_legend();
a = sns.FacetGrid(df_train, row='Embarked')

a.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')

a.add_legend();
print('Before.', df_train.shape, df_test.shape, comb[0].shape, comb[1].shape)

df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)

df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)

comb = [df_train, df_test]

print('Later.', df_train.shape, df_test.shape, comb[0].shape, comb[1].shape)
for dataset in comb:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(df_train.Title, df_train.Sex)
for dataset in comb:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
df_train = df_train.drop(['Name', 'PassengerId'], axis=1)

df_test = df_test.drop(['Name'], axis=1)

combine = ['df_train', 'df_test']

df_train.shape, df_test.shape
sex = pd.get_dummies(df_train['Sex'],drop_first=True)

sex1 = pd.get_dummies(df_test['Sex'],drop_first=True)

embark = pd.get_dummies(df_train['Embarked'],drop_first=True)

embark1 = pd.get_dummies(df_test['Embarked'],drop_first=True)
df_train.drop(['Sex','Embarked'], axis=1, inplace=True)

df_test.drop(['Sex','Embarked'], axis=1, inplace=True)

train = pd.concat([df_train,sex,embark],axis=1)

test = pd.concat([df_test,sex1,embark1],axis=1)
train.head()
test.head()
combine = ['train','test']
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
combine = ['train', 'test']
train.info()
train['AgeBand'] = pd.cut(train['Age'], 5)

train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
train.head()
train.loc[ train['Age'] <= 16, 'Age'] = 0

train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1

train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2

train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3

train.loc[ train['Age'] > 64, 'Age']

train.head()
test.loc[ test['Age'] <= 16, 'Age'] = 0

test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1

test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2

test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3

test.loc[ test['Age'] > 64, 'Age']

test.head()
train = train.drop(['AgeBand'], axis=1)

train.head()
combine = ['train', 'test']
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1



train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1



test.head()
train['IsAlone'] = 0

train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1



train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
test['IsAlone'] = 0

test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train, test]



train.head()
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

test.head()
train['Fareband'] = pd.qcut(train['Fare'], 4)

train[['Fareband', 'Survived']].groupby(['Fareband'], as_index=False).mean().sort_values(by='Fareband', ascending=True)
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0

train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1

train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2

train.loc[ train['Fare'] > 31, 'Fare'] = 3

train['Fare'] = train['Fare'].astype(int)



train = train.drop(['Fareband'], axis=1)

train.head()
test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0

test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1

test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2

test.loc[ test['Fare'] > 31, 'Fare'] = 3

test['Fare'] = test['Fare'].astype(int)
test.head()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

train['Title'] = train['Title'].map(title_mapping)

train['Title'] = train['Title'].fillna(0)

test['Title'] = test['Title'].map(title_mapping)

test['Title'] = test['Title'].fillna(0)





test.head()
train.head()
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred1 = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)