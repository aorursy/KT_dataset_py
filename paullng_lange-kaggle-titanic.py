# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
input_path = "../input/titanic/"

df_train = pd.read_csv(input_path+'train.csv')

df_test = pd.read_csv(input_path+'test.csv')
df_train
sns.catplot(x="Embarked", y="Survived", kind="bar" , data=df_train);
southampton = sum(df_train["Embarked"] == "S")

cherbourg = sum(df_train["Embarked"] == "C")

queenstown = sum(df_train["Embarked"] == "Q")





print("Nombre de personne ayant pris la porte d'embarquement S (Southampton):")

print(southampton)
print("Nombre de personne ayant pris la porte d'embarquement C (cherbourg):")

print(cherbourg)
print("Nombre de personne ayant pris la porte d'embarquement Q (queenstown):")

print(queenstown)
sns.catplot(x="SibSp", y="Survived", kind="bar" , data=df_train);
sns.catplot(x="Pclass", y="Survived", kind="bar" , data=df_train);
sns.catplot(x="Sex", y="Survived", kind="bar", data=df_train);
df_train = df_train.dropna(subset=['Age'])

df_train
df_alive = df_train.loc[df_train['Survived'] == 1]

df_died = df_train.loc[df_train['Survived'] == 0]
df_alive
df_died
moyenneAgeVivant = sum(df_alive['Age'])/len(df_alive)

moyenneAgeVivant
moyenneAgeMort = sum(df_died['Age'])/len(df_died)

moyenneAgeMort
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100)
X_train = df_train.loc[:, ['Pclass', 'SibSp']]
y_train = df_train['Survived']
X_test = df_test.loc[:, ['Pclass', 'SibSp']]
from sklearn.model_selection import train_test_split

def train_model(model, X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))

    test_acc = accuracy_score(y_test, model.predict(X_test))

    return {

        'train accuracy': train_acc,

        'test accuracy': test_acc

    }

print(train_model(rf, X_train, y_train))
rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)