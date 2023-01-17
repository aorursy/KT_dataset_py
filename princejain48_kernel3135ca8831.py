# -*- coding: utf-8 -*-

"""

Created on Wed Feb 13 22:11:55 2019



@author: Shree

"""

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score, precision_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

import seaborn as sns

import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

train.head()

words = ["Mr.", "Mrs.", "Miss.", "Capt.","Col.", "Major.", "Jonkheer.", "Don.", 'Sir.', "Dr.", "Rev.", "the Countess.", "Dona", "Mme", "Mlle", "Ms",  "Master", "Lady"]

train['Name'] = [' '.join(w for w in t.split() if w in words) for t in train['Name']]

test['Name'] = [' '.join(w for w in t.split() if w in words) for t in test['Name']]

Title_Dictionary = {

                        "Capt.":       "Officer",

                        "Col.":        "Officer",

                        "Major.":      "Officer",

                        "Jonkheer.":   "Royalty",

                        "Don.":        "Royalty",

                        "Sir." :       "Royalty",

                        "Dr.":         "Officer",

                        "Rev.":        "Officer",

                        "the Countess.":"Royalty",

                        "Dona.":       "Royalty",

                        "Mme.":        "Mrs",

                        "Mlle.":       "Miss",

                        "Ms.":         "Mrs",

                        "Mr." :        "Mr",

                        "Mrs." :       "Mrs",

                        "Miss." :      "Miss",

                        "Master." :    "Master",

                        "Lady." :      "Royalty"



                        }

newfeature= train['Name'].map(Title_Dictionary)

newfeature_test = test['Name'].map(Title_Dictionary)

newfeature.value_counts()

sns.barplot(x=newfeature, y ='Survived', data=train)

titles_dummy = pd.get_dummies(newfeature, prefix='Title')

train = pd.concat([train, titles_dummy], axis=1)

titles_dummy_test = pd.get_dummies(newfeature_test, prefix='Title')

test = pd.concat([test, titles_dummy_test], axis=1)

train.drop('Name',axis=1,inplace=True)

test.drop('Name',axis=1,inplace=True)

train.head()

train['Sex'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)

test['Sex'] = test['Sex'].map({'female': 0, 'male': 1}).astype(int)

test['Sex'].value_counts()

fig, axis1= plt.subplots(figsize=(8,3))

sns.countplot(x='Sex', data=train, ax=axis1)

embark = train['Embarked'].fillna('S')

train['Embarked'] = embark.map({'S': 1, 'C': 2, 'Q': 3}).astype(int)

test['Embarked'] = embark.map({'S': 1, 'C': 2, 'Q': 3}).astype(int)

sns.barplot(x='Embarked', y ='Survived', data=train)

train['Age'] = train['Age'].fillna(train['Age'].median()).astype(int)

test['Age'] = test['Age'].fillna(train['Age'].median()).astype(int)

Embarked_dummy = pd.get_dummies(train['Embarked'], prefix= 'Embarked')

train = pd.concat([train, Embarked_dummy], axis= 1)

Embarked_dummy_test = pd.get_dummies(test['Embarked'], prefix= 'Embarked')

test = pd.concat([test, Embarked_dummy_test], axis= 1)

train.drop('Embarked', axis=1,inplace=True)

test.drop('Embarked', axis=1,inplace=True)

train.head()

train['Relativesinship'] = train['SibSp'] + train['Parch']

test['Relativesinship'] = test['SibSp'] + test['Parch']

train[['Relativesinship', 'Survived']].groupby(['Relativesinship']).mean()

test['Fare'].fillna(test['Fare'].median(), inplace = True)

train['Fare'] = train['Fare'].astype(int)

test['Fare'] = test['Fare'].astype(int)

train.drop('Ticket',axis=1,inplace=True)

train.drop('Cabin',axis=1,inplace=True)

def age_cat(age):

    if age <= 16:

        return 0

    elif 16< age <=26:

        return 1

    elif 26< age <=36:

        return 2

    elif 36< age <=47:

        return 3

    elif 47 < age:

        return 4

    

train['Age'] = train['Age'].apply(age_cat)

test['Age'] = test['Age'].apply(age_cat)

Age_dummy = pd.get_dummies(train['Age'], prefix= 'Age')

train = pd.concat([train, Age_dummy], axis= 1)

Age_dummy_test = pd.get_dummies(test['Age'], prefix= 'Age')

test = pd.concat([test, Age_dummy_test], axis= 1)

train.drop('Age', axis=1, inplace=True)

test.drop('Age', axis=1, inplace=True)

train.head()

X_train = train[['Pclass','Sex', 'Age_0','Age_1','Age_2', 'Age_3', 'Age_4', 'Relativesinship', 'Fare','Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Embarked_1', 'Embarked_2', 'Embarked_3']]

y_train = train[['Survived']]

X_test = test[['Pclass','Sex', 'Age_0', 'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Relativesinship', 'Fare','Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Embarked_1', 'Embarked_2', 'Embarked_3']]



columns = ['Pclass','Sex', 'Age_0','Age_1','Age_2', 'Age_3', 'Age_4', 'Relativesinship', 'Fare','Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Embarked_1', 'Embarked_2', 'Embarked_3']

X_train = X_train .reindex(columns= columns)

X_test = X_test.reindex(columns= columns)



X_train[columns] = X_train[columns].astype(int)

X_test[columns] = X_test[columns].astype(int)

from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(max_iter=5, random_state=42)

sgd_clf.fit(X_train, y_train)

cross_clf_score = cross_val_score(sgd_clf, X_train, y_train, cv = 10, scoring = 'accuracy')

cross_clf_score.mean()

y_train_clf_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

print(precision_score(y_train, y_train_clf_pred ))

print(recall_score(y_train, y_train_clf_pred ))

conf_mx = confusion_matrix(y_train, y_train_clf_pred )

plt.matshow(conf_mx, cmap=plt.cm.Blues)

from sklearn.metrics import f1_score

f1_score(y_train, y_train_clf_pred)

from sklearn.metrics import roc_auc_score

roc_auc_score(y_train, y_train_clf_pred)