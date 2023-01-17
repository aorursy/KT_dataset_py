# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # graph

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading dataset

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
print(train.shape)

print(test.shape)
train.head()
train.columns
# Survived reparition

sns.countplot(train['Survived'])
train.Survived.value_counts()
# Survivied reparition

sns.countplot(train['Pclass'])
train.Pclass.value_counts()
sns.catplot(x="Pclass", y="Survived", kind="bar", data=train)
train[['Pclass', 'Survived']].groupby('Survived').mean()
sns.countplot(train.Sex)
sns.catplot(x="Sex", y="Survived", kind="bar", data=train)
sns.catplot(x="Pclass", y="Survived", hue="Sex", kind="bar", data=train)
train.Age.isnull().sum()
train.Age = train.Age.fillna(np.mean(train.Age))
train.Age.describe()
sns.set(rc={'figure.figsize':(9,7)})

sns.distplot(train.Age)
grid = sns.FacetGrid(train, row='Survived', size=3, aspect=1.6)

grid.map(sns.distplot, 'Age', 'Survived')

grid.add_legend()
sns.distplot(train[train.Survived == 0].Age)
print(round(np.mean(train[train.Survived == 1].Age),1))

print(round(np.mean(train[train.Survived == 0].Age),1))
sns.countplot(train.Parch)
sns.catplot(x="Parch", y="Survived", kind="bar", data=train)
pd.crosstab(train.Parch, train.Survived)
sns.countplot(x="Parch", hue ="Survived", data=train)
train[train.Parch >= 4]
train[train.Parch >= 4][['Parch', 'Survived']].groupby('Parch').mean()
sns.catplot(x="Parch", y="Survived", hue="Sex", kind="bar", data=train)
sns.distplot(train.Fare)
sns.distplot(train[train.Fare <100].Fare)
np.round(len(train[train.Fare < 100]) / len(train),2)
train.Fare.describe()
train[["Fare", "Survived"]].groupby('Survived').mean()
train[["Fare", "Survived", "Pclass"]].groupby('Survived').mean()
train[["Fare", "Pclass"]].groupby('Pclass').mean()
sns.countplot(train.Embarked)
sns.catplot(x="Embarked", y="Survived", kind="bar", data=train)
sns.countplot(x="Embarked", hue ="Survived", data=train)
train[['PassengerId', 'Survived','Embarked']].groupby(['Survived','Embarked']).count()
train[['Survived','Embarked', 'Fare']].groupby(['Survived','Embarked']).mean()
train[['Embarked', 'Pclass']].groupby(['Embarked']).mean()
train[['Embarked', 'Fare']].groupby(['Embarked']).mean()
train[['Survived','Embarked', 'Pclass']].groupby(['Survived','Embarked']).mean()
sns.catplot(x="Embarked", y="Survived", hue="Pclass", kind="bar", data=train)
grid = sns.FacetGrid(train, row='Embarked', size=3, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')

grid.add_legend()
# Reading dataset

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train_len = len(train)
# Regroup train and test in data

data = train.append(test)
data.shape
data.isnull().sum()
data.head()
age = data.dropna()
age.Age.isnull().sum()
age[['Age', 'Sex']].groupby('Sex').mean().round(0)
age[['Age', 'Pclass']].groupby('Pclass').mean().round(0)
pclass = data.Pclass.unique()



for i, row in data.iterrows():

    for p in pclass:

        data.loc[(data.Pclass == p) & (data.Age.isnull()), 'Age'] = np.mean(data[data.Pclass == p].Age)
data['Age'] = np.round(data['Age'],0)
data.isnull().sum()
data[data.Fare.isnull()]
data[['Pclass', 'Fare']].groupby('Pclass').mean()
data[data.Pclass == 3].Fare.mean()
data[['Embarked', 'Fare']].groupby('Embarked').mean()
data[(data.Pclass == 3) & (data.Embarked == 'S')].Fare.mean()
data[['Parch', 'Fare']].groupby('Parch').mean()
data[(data.Pclass == 3) & (data.Parch == 0)].Fare.mean()
data[['Age', 'Fare']].groupby('Age').mean()
data[(data.Age > 50) & (data.Pclass == 3)].Fare.mean()
data[data.Fare.isnull()]
data.loc[data.Fare.isnull(), 'Fare'] = data[(data.Pclass == 3) & (data.Embarked == 'S') & (data.Parch == 0)].Fare.mean()
data.loc[152]
data.isnull().sum()
data[data.Embarked.isnull()]
data[['PassengerId', 'Pclass','Embarked']].groupby(['Pclass','Embarked']).count()
tickets = data[data.Ticket.str.startswith('1135')]
tickets[['PassengerId', 'Pclass','Embarked']].groupby(['Pclass','Embarked']).count()
tickets
data.loc[data.Embarked.isnull(), 'Embarked'] = 'C'
data.isnull().sum()
data[data.Cabin.notna()].head(30)
print("Missing cabins : " , round(data.Cabin.isnull().sum() / len(data),2),"%")
data.Cabin = data.Cabin.fillna("X")
data.Cabin = [i[0] for i in data.Cabin]
data.Cabin.value_counts()
data.groupby('Cabin')['Fare', 'Pclass'].mean().sort_values(by='Fare')
(107+122)/2
data.loc[(data['Fare'] < 16) & (data['Cabin'] == 'X'), 'Cabin'] = 'G'

data.loc[(data['Fare'] >= 16) & (data['Fare'] < 27) & (data['Cabin'] == 'X'), 'Cabin'] = 'F'

data.loc[(data['Fare'] >= 27) & (data['Fare'] < 38) & (data['Cabin'] == 'X'), 'Cabin'] = 'T'

data.loc[(data['Fare'] >= 38) & (data['Fare'] < 47) & (data['Cabin'] == 'X'), 'Cabin'] = 'A'

data.loc[(data['Fare'] >= 47) & (data['Fare'] < 53) & (data['Cabin'] == 'X'), 'Cabin'] = 'D'

data.loc[(data['Fare'] >= 53) & (data['Fare'] < 54) & (data['Cabin'] == 'X'), 'Cabin'] = 'E'

data.loc[(data['Fare'] >= 54) & (data['Fare'] < 115) & (data['Cabin'] == 'X'), 'Cabin'] = 'C'

data.loc[(data['Fare'] > 115) & (data['Cabin'] == 'X'), 'Cabin'] = 'B'
data.Cabin.value_counts()
data.isnull().sum()
# Title



data['Title'] = [i.split(', ')[1] for i in data['Name']]
data['Title'] = [i.split('.')[0] for i in data['Title']]
data.Title.value_counts()
# We replace the rare titles into a new category



data["Title"] = [i.replace('Ms', 'Miss') for i in data.Title]

data["Title"] = [i.replace('Mlle', 'Miss') for i in data.Title]

data["Title"] = [i.replace('Mme', 'Mrs') for i in data.Title]



data["Title"] = [i.replace('Dr', 'others') for i in data.Title]

data["Title"] = [i.replace('Rev', 'others') for i in data.Title]

data["Title"] = [i.replace('Col', 'others') for i in data.Title]

data["Title"] = [i.replace('Major', 'others') for i in data.Title]

data["Title"] = [i.replace('Capt', 'others') for i in data.Title]

data["Title"] = [i.replace('Dona', 'others') for i in data.Title]

data["Title"] = [i.replace('Jonkheer', 'others') for i in data.Title]

data["Title"] = [i.replace('Don', 'others') for i in data.Title]

data["Title"] = [i.replace('Sir', 'others') for i in data.Title]

data["Title"] = [i.replace('the Countess', 'others') for i in data.Title]

data["Title"] = [i.replace('Lady', 'others') for i in data.Title]
data.Title.value_counts()
data.head()
# Family size



data['family_size'] = data['Parch'] + data['SibSp'] + 1 # +1 for the person
# Is alone



data['is_alone'] = [1 if i < 2 else 0 for i in data.family_size]
data['Sex'] = data['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q':2} ).astype(int)
data['Cabin'] = data['Cabin'].map( {'G': 0, 'F': 1, 'T': 2, 'A': 3, 'D': 4, 'E': 5, 'C': 6, 'B': 7} ).astype(int)
data['Title'] = data['Title'].map( {'Mr': 0, 'Miss': 1, 'Mrs':2, 'Master':3, 'others':4} ).astype(int)
data.size
sns.distplot(data.Age, bins=8)
train['AgeBin'] = pd.cut(train['Age'], 8)

train[['AgeBin', 'Survived']].groupby(['AgeBin'], as_index=False).mean().sort_values(by='AgeBin', ascending=True)
data.loc[ data['Age'] <= 10, 'Age'] = 0

data.loc[(data['Age'] > 10) & (data['Age'] <= 20), 'Age'] = 1

data.loc[(data['Age'] > 20) & (data['Age'] <= 30), 'Age'] = 2

data.loc[(data['Age'] > 30) & (data['Age'] <= 40), 'Age'] = 3

data.loc[(data['Age'] > 40) & (data['Age'] <= 50), 'Age'] = 4

data.loc[(data['Age'] > 50) & (data['Age'] <= 60), 'Age'] = 5

data.loc[(data['Age'] > 60) & (data['Age'] <= 70), 'Age'] = 6

data.loc[ data['Age'] > 70, 'Age'] = 7
train.Fare.describe()
data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0

data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.45), 'Fare'] = 1

data.loc[(data['Fare'] > 14.45) & (data['Fare'] <= 31), 'Fare'] = 2

data.loc[data['Fare'] > 31, 'Fare'] = 3
data.head()
data = data.drop(['Ticket', 'Name'], axis=1)
data.Age = data.Age.astype(int)

data.Fare = data.Fare.astype(int)
train = data[:train_len]

test = data[train_len:]
train.head()
test.head()
X_train = train.drop(['Survived', 'PassengerId'], axis=1)

y_train = train['Survived']

X_test = test.drop(['Survived','PassengerId'], axis=1)
data.dtypes
train.corr()['Survived'].sort_values(ascending=False)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC, LinearSVC

from xgboost import XGBClassifier
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log
# KNN

from sklearn.model_selection import cross_val_score



neighbors = range(1,7)

cv_scores = []



for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())



# changing to misclassification error

MSE = [1 - x for x in cv_scores]



# determining best k

optimal_k = neighbors[MSE.index(min(MSE))]



print(optimal_k)
# plot misclassification error vs k

plt.plot(neighbors, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()
knn = KNeighborsClassifier(n_neighbors = optimal_k)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
# Support Vector Machines



svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)

acc_svc
# XGBoost



xgb = XGBClassifier()

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

acc_xgb = round(xgb.score(X_train, y_train), 2)

print(acc_xgb)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
y_pred = y_pred.astype(int)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('submission.csv', index=False)