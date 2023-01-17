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
df_train.info()
df_test.info()
df_train.isnull().sum()
df_train['age_is_null'] = np.where(df_train['Age'].isnull() , 1, 0);

df_test['age_is_null'] = np.where(df_test['Age'].isnull() , 1, 0);
sns.catplot(x="age_is_null", y="Survived", kind="bar", data=df_train);
df_train['cabin_is_null'] = np.where(df_train['Cabin'].isnull() , 1, 0)

df_test['cabin_is_null'] = np.where(df_test['Cabin'].isnull() , 1, 0)
sns.barplot(x="cabin_is_null", y="Survived", data=df_train)

plt.show()
X_train = df_train.loc[:, ['Pclass', 'SibSp']]
y_train = df_train['Survived']
X_test = df_test.loc[:, ['Pclass', 'SibSp']]
sns.catplot(x="Sex", y="Survived", kind="bar", data=df_train);
sns.catplot(x="Pclass", y="Survived", kind="bar", data=df_train);
sns.catplot(x="SibSp", y="Survived", kind="bar", data=df_train);
sns.catplot(x="Embarked", y="Survived", kind="bar", data=df_train);



print("Nombre de personne qui ont embarqué à Southampton (S):")

southampton = df_train[df_train["Embarked"] == "S"].shape[0]

print(southampton)



print("Nombre de personne qui ont embarqué à Cherbourg (C):")

cherbourg = df_train[df_train["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Nombre de personne qui ont embarqué à Queenstown (Q):")

queenstown = df_train[df_train["Embarked"] == "Q"].shape[0]

print(queenstown)
# On remplace les valeurs manquantes de Embarked par S

df_train['Embarked'] = df_train['Embarked'].fillna('S')

df_test['Embarked'] = df_test['Embarked'].fillna('S')
# On convertit les valeurs en int

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

df_train['Embarked'] = df_train['Embarked'].map(embarked_mapping)

df_test['Embarked'] = df_test['Embarked'].map(embarked_mapping)
embarked_mapping = {"female": 1, "male": 0}

df_train['Sex'] = df_train['Sex'].map(embarked_mapping)

df_test['Sex'] = df_test['Sex'].map(embarked_mapping)



df_train.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()
df_test.isna().sum() 
my_cols = ['Pclass', 'SibSp', 'Embarked', 'Sex', 'age_is_null', 'cabin_is_null']
X_train = df_train.loc[:, my_cols]
X_train
y_train = df_train['Survived']
X_test = df_test.loc[:, my_cols]
X_train.info()
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