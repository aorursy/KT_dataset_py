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
df_train.describe()
df_train.head()
df_train.tail()
df_train.info()
df_train.isna().sum()
df_test.isna().sum()
df_test.describe()
# on affiche la moyenne de l'age

moyenne_age = np.sum(df_train['Age'])/len(df_train.loc[df_train['Age']>0])

moyenne_age

#on remplace les ages par la moyenne

df_train['Age'] = df_train['Age'].fillna(moyenne_age)

df_test['Age'] = df_test['Age'].fillna(moyenne_age)

df_train
# on attribue à homme et femme un boolean

embarked_mapping = {"female": 1, "male": 0}

df_train['Sex'] = df_train['Sex'].map(embarked_mapping)

df_test['Sex'] = df_test['Sex'].map(embarked_mapping)



df_train.head()
# on affiche la moyenne du Fare

moyenne_fare = np.sum(df_train['Fare'])/len(df_train.loc[df_train['Fare']>0])

moyenne_fare



# on remplace le fare par la moyenne

df_train['Fare'] = df_train['Fare'].fillna(moyenne_fare)

df_test['Fare'] = df_test['Fare'].fillna(moyenne_fare)

df_train
# nombre de personnes dans la famille

df_train['famille'] = df_train['SibSp'] + df_train['Parch']

df_test['famille'] = df_test['SibSp'] + df_test['Parch']

df_train[['Survived', 'famille']]
sns.barplot(x='famille', y="Survived", data=df_train)
sns.boxplot(x='Survived', y="Fare", data=df_train.loc[df_train['Fare'] <500]);
sns.barplot(x='Pclass', y="Survived", data=df_train)
sns.catplot(x="Sex", y="Survived", kind="bar", data=df_train);
# ajout de colonnes

modele = ['Age', 'Sex', 'Pclass', 'Fare', 'SibSp', 'Parch']



X_train = df_train.loc[:, modele]



y_train = df_train['Survived']



X_test = df_test.loc[:, modele]
X_train.info()
X_test.info()
X_train.isna().sum()
X_test.isna().sum()
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
submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)