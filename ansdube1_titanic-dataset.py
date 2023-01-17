import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
train_data = pd.read_csv("../input/train.csv", index_col='PassengerId')

test_data = pd.read_csv("../input/test.csv", index_col='PassengerId')
test_data["Survived"] = 999

combined_data = pd.concat((train_data, test_data), axis =0)
combined_data.info()
combined_data.head()
sns.countplot(x='Survived', data=combined_data)
sns.countplot(x='Survived', hue='Sex', data=combined_data)
sns.countplot(x='Survived', hue='Pclass', data=combined_data)
sns.countplot(x='SibSp', data=combined_data)
combined_data.Age.hist(bins=20)
sns.heatmap(combined_data.isnull())
combined_data.drop('Cabin', axis=1, inplace=True)
combined_data[combined_data.Fare.isnull()]
combined_data[combined_data.Embarked.isnull()]
combined_data.pivot_table(index='Embarked', columns='Pclass', values='Fare', aggfunc='mean')
combined_data.pivot_table(index='Embarked', columns='Pclass', values='Fare', aggfunc='median')
combined_data.Embarked.fillna('C', inplace=True)
combined_data.Fare.fillna('8.05', inplace=True)
combined_data.boxplot('Age', 'Sex')
def getTitle(name):

    name_after_split = name.split(', ')[1]

    title = name_after_split.split('.')[0]

    title = title.strip().lower()

    return title
combined_data["Title"] = combined_data.Name.map(lambda x : getTitle(x))
combined_data.head(10)
combined_data.info()
title_median_age = combined_data.groupby('Title').Age.transform('median')
combined_data.Age.fillna(title_median_age, inplace=True)
sns.heatmap(combined_data.isnull())
combined_data['FamilySize'] = combined_data.Parch + combined_data.SibSp
def getMaturity(age):

    if age < 18:

        return 'CHILD'

    elif age >= 18 and age <=40:

        return 'ADULT'

    else:

        return 'OLD'
combined_data['Maturity'] = combined_data.Age.map(lambda x : getMaturity(x))
combined_data.head(10)
combined_data.drop("Parch", axis=1, inplace=True)

combined_data.drop("SibSp", axis=1, inplace=True)

combined_data.drop("Ticket", axis=1, inplace=True)

combined_data.drop("Title", axis=1, inplace=True)

combined_data.drop("Name", axis=1, inplace=True)
combined_data.info()
sns.countplot(x='Survived', hue='Maturity', data=combined_data)
combined_data.head(10)
combined_data.info()
combined_data.Fare = combined_data.Fare.astype('float', inplace=True)
combined_data.info()
combined_data.Fare.plot(kind='box')
combined_data["Sex"] = pd.Categorical(combined_data["Sex"])

combined_data["Embarked"] = pd.Categorical(combined_data["Embarked"])

combined_data["Maturity"] = pd.Categorical(combined_data["Maturity"])

combined_data["Pclass"] = pd.Categorical(combined_data["Pclass"])
combined_data = pd.get_dummies(combined_data, drop_first=True)
combined_data.head(10)
train_data = combined_data[combined_data.Survived != 999]

test_data = combined_data[combined_data.Survived == 999][combined_data.loc[:, combined_data.columns != "Survived"].columns]

train_data.shape #(891, 11)

combined_data.shape

test_data.info()
X = train_data.drop("Survived", axis=1)

Y = train_data['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
test_predicitions = logreg.predict(X_test)
classification_report(Y_test, test_predicitions)
confusion_matrix(Y_test, test_predicitions)
accuracy_score(Y_test, test_predicitions)