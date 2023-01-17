import warnings

warnings.filterwarnings("ignore")



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

data = train.append(test, ignore_index=True)
train.describe()
train.isnull().sum()
test.isnull().sum()
pd.DataFrame(train.groupby(['Sex']).mean())
s = sns.barplot(x='Sex', y='Survived', data=train)
pd.DataFrame(train.groupby(['Pclass']).mean())
s = sns.barplot(x='Pclass', y='Survived', data=train)
plt.figure(figsize=(10,6))

s = sns.boxplot(x='Pclass', y='Age', data=train)
pd.DataFrame(train.groupby(['SibSp']).mean())
s = sns.factorplot(x='SibSp', y='Survived', data=train, palette='dark', height=6, aspect=2)
plt.figure(figsize=(14,6))

s = sns.boxplot(x='SibSp', y='Age',data=train)
pd.DataFrame(train.groupby(['Parch']).mean())
s = sns.factorplot(x='Parch', y='Survived', data=train, height=6, aspect=2, palette='dark')
s = sns.distplot(train['Fare'])
train[train.Fare>500]
pd.DataFrame(train.groupby(['Embarked']).mean())
s = sns.barplot(x='Embarked', y='Survived', data=train)
s = sns.distplot(train['Age'])
train.corr().Age
train.Name.head()
train[train.Cabin.isnull()==False].Cabin.head()
train.Ticket.head()
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
s = sns.factorplot(x='FamilySize', y='Survived', data=data, palette='dark', height=6, aspect=2)
ticket_count = data.Ticket.value_counts()

ticket_count
for i in ticket_count.index:

    data.loc[data.Ticket.str[:]==i, 'IsAlone'] = (ticket_count[i] == 1)
data['Title'] = data.Name.str.extract('([a-zA-z]+)\.')
data.Title.value_counts()
s = sns.factorplot(x='Title', y='Survived', data=data, palette='dark', height=6, aspect=2)
s = sns.factorplot(x='Title', y='Age', data=data, palette='dark', height=6, aspect=2)
title_age = data.groupby('Title').Age.mean()
for i in title_age.index:

    data.loc[(data.Title.str[:]==i) & (data.Age.isnull()==True), 'Age'] = title_age[i].astype(float)
plt.figure(figsize=(14, 7))

s = sns.violinplot(x='Survived', y='Age', data=data)

plt.ylim(0, 90)

plt.show()
bins = [0,12,24,45,60, data.Age.max()] 

labels = ['Child', 'Teenager', 'Adult','Senior','Old'] 

data["Age"] = pd.cut(data["Age"], bins, labels = labels)
data.Age.value_counts()
data.Embarked.value_counts()
data.Embarked.fillna('S', inplace=True)
data['Cabin_letter'] = data['Cabin'].str[0]
data.groupby('Pclass').Cabin_letter.value_counts()
s = sns.factorplot(x='Cabin_letter', y='Pclass', data=data, palette='dark', height=6, aspect=2)
data.loc[(data.Pclass==1) & (data['Cabin_letter'].isnull()==True), 'Cabin_letter'] = 'C'

data.loc[(data.Pclass==2) & (data['Cabin_letter'].isnull()==True), 'Cabin_letter'] = 'D'

data.loc[(data.Pclass==3) & (data['Cabin_letter'].isnull()==True), 'Cabin_letter'] = 'G'
s = sns.factorplot(x='Cabin_letter', y='Survived', data=data, palette='dark', height=6, aspect=2)
data = data.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Fare'], axis=1)
data.head()
data = pd.get_dummies(data)
trainSet = data.iloc[:891]

testSet = data.iloc[891:]

y = trainSet['Survived']

X = trainSet.drop('Survived', axis=1)

testX = testSet.drop('Survived', axis=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
def score(n_estimators, max_depth):

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    scores = cross_val_score(model, X, y, cv=10, scoring='f1', n_jobs=-1)

    return scores.mean()
# estimators = [100, 500, 1000, 1500, 2000]

# depths = [3, 4, 5]

# for i in estimators:

#     for j in depths:

#         print("estimators: {}, depth: {}, score: {}".format(i, j, score(i, j)))
model = RandomForestClassifier(n_estimators=1000, max_depth=4)
model.fit(X, y)

pred = model.predict(testX).astype(int)

ans = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred})

ans.to_csv("submission.csv", index=False)

print('done')