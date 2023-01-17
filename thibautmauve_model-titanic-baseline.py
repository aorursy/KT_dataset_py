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
df_train.info()
df_train
df_train.isna().sum()
df_train.describe()
# Ici je flag les valeurs de la donnée "Age" qui sont nulles et qui seront remplacées par la suite : si l'âge est nul la valeur d'"Age_is_null" sera 1, sinon 0

df_train['Age_is_null'] = np.where(df_train['Age'].isnull(), 1, 0)
df_train
# On reproduit le même traitement sur le set de test

df_test['Age_is_null'] = np.where(df_test['Age'].isnull(), 1, 0)
df_test
# Ici je remplace les âges manquants par l'âge médian (cf Out[45]).

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_train['Age']
# La même chose sur le set de test

df_test['Age'] = df_test['Age'].fillna(df_train['Age'].median())
df_test['Age']
sns.catplot(x="Sex", y="Age", kind="bar", data=df_train, height=4, aspect=2);
sns.catplot(x="Sex", y="Survived", kind="bar", data=df_train, height=4, aspect=2);
# Comme pour l'âge, je flag les valeurs nulles de la donnée "Cabin" : 1 = cabine nulle, 0 = cabine retrouvée.

df_train['Cabin_is_null'] = np.where(df_train['Cabin'].isnull(), 1, 0)
df_train
# Même chose pour le DF de test

df_test['Cabin_is_null'] = np.where(df_test['Cabin'].isnull(), 1, 0)
df_test
sns.catplot(x="Cabin_is_null", y="Survived", kind="bar", data=df_train, height=4, aspect=2);
sns.catplot(x="Pclass", y="Survived", kind="bar", data=df_train, height=4, aspect=2);
plt.figure(figsize=(20,10))

sns.distplot(df_train["Fare"]);
plt.figure(figsize=(20,10))

sns.distplot(df_train["Pclass"]);
sns.barplot(x='Embarked', y='Survived', data=df_train);
# D'abord sur le set de train

df_train['Embarked'] = df_train['Embarked'].map({'S': 0, 'C': 1, 'Q':2})
# Puis sur le set de test

df_test['Embarked'] = df_test['Embarked'].map({'S': 0, 'C': 1, 'Q':2})
sns.barplot(x='SibSp', y='Survived', data=df_train);
sns.barplot(x='Parch', y='Survived', data=df_train);
NUM_FEATURES = ['Age', 'Pclass', 'Fare', 'SibSp', 'Cabin_is_null', 'Parch']

CAT_FEATURES = ['Sex', 'Embarked']

my_cols = NUM_FEATURES + CAT_FEATURES
my_cols
# On fait correspondre les valeurs de "Sex" à des valeurs numériques pour pouvoir construire le modèle.

df_train['Sex'] = df_train['Sex'].map({'male': 0, 'female': 1})
# La même chose sur la donnée "Age" du set de test

df_test['Sex']= df_test['Sex'].map({'male': 0, 'female': 1})
# 0.0 est la valeur la plus fréquente, aussi on remplace les valeurs nulles de 'Embarked' par cette valeur.

df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])
X_train = df_train.loc[:,my_cols]
y_train = df_train['Survived']
df_test['Fare'] = df_test['Fare'].fillna(df_train['Fare'].median())
X_test = df_test.loc[:,my_cols]
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
# La cellule suivante a pour objectif de démontrer l'importance de chaque donnée dans le degré de précision du modèle. Dans l'ordre :

# On définit une variable X de la même façon que le set de train et le set de test.

X = df_test.loc[:,my_cols]

# On crée un dictionnaire qui contient les colonnes sélectionnées pour le modèle et leurs degrés d'importance

fi_dict = {

    'feats': X.columns,

    'feature_importance': rf.feature_importances_

}



# On crée un DF à partir de ce dictionnaire. Les données sont triées par ordre d'importance de la moins importante à la plus importante.

fi = pd.DataFrame(fi_dict).set_index('feats').sort_values(

    'feature_importance', ascending=False)



# On affiche les données et leurs importances dans l'ordre décroissant, de la plus importante à la moins importante.

fi.sort_values(

    'feature_importance', ascending=True).tail(10).plot.barh();
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)