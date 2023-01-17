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
#Par défaut, nous nous retrouvons dans le dossier output, c'est pourquoi il est neccésaire de remonter d'un étage afin d'accéder au dossier input

input_path = "../input/titanic/"

df_train = pd.read_csv(input_path+'train.csv')

df_test = pd.read_csv(input_path+'test.csv')
# On appelle le dataframe directement afin d'avoir les 5 premières et dernières lignes, avec cela, nous pouvons avoir une idée global de la qualité du dataset et des actions nécessaire à son amélioration.

df_train
# La même chose pour celui de test

df_test
# Afin d'avoir le plus de donnée possible pour réaliser des moyennes et autres, j'ai decidé de concatener les deux dataframes mais pour ce faire, j'ai besoin qu'il ait le même nombre de champs, je complete avec -1 car cela ma permetera de savoir si ce sont des données de test ou non

df_test['Survived'] = -1
# On concatène les deux tableaux en lui demandant de ne pas les trier.

df = pd.concat([df_train, df_test], sort=False)
# Nous avons bien les "-1" ce qui montre que test a été concaténé à la suite du dataframe train

df
# Nous allons regarder les types de chaque champs

df.info()

# Comme nous le voyons, il y a beaucoup de type 'object', object represente une donnée mais ce type est difficile d'interpretation pour la machine et notre modèle, c'est pourquoi nous devons transformer un maximum de champs en valeur numérique
# On va regarder les colonnes qui ont des valeurs manquantes et comme on peux le voir, il y a la colonne Age, Cabin et Embarked. On peux dire que le dataset est plutot de bonne qualité

df.isna().sum()
df.describe()
# Commençons avec l'age

# Nous allons remplacer les valeurs null par la moyenne de cette colonne

df['Age'].fillna( np.sum(df['Age'])//len(df.loc[df['Age'] > 0]) , inplace=True)

# Fillna permet de remplacer les valeurs null par une valeur donnée en premier paramètre

# Inplace permet de remplacer le df directement et ne pas être obliger de l'assigner à une nouvelle variable

# Le // permet de nous retourner un integer et non un float

# En premier paramètre nous lui donnons la moyenne de la colonne en ne divisant que par le nombre de ligne qui contient  un age valide
# Regardons si la modification a bien été importé

df.isna().sum()

# Nous voyons bien qu'il n'y a plus de valeur null sur la colonne age
# Maintenant, nous allons remplacer les colonnes de type object par un type numérique.

# Pour cela, nous avons besoin de savoir quelles sont les valeurs de cette colonne.

df['Sex'].value_counts()

# Comme nous pouvons voir, il s'agit de 'male' et 'female', nous allons les transformer en valeur numérique à savoir 0 et 1
df['Sex'] = df['Sex'].map({

        'male': 0, 

        'female': 1,

}).astype(int)

df.info()

# Nous avons remplacer le sex par une valeur numérique
sns.catplot(x="Sex", y="Survived", kind="bar", data=df.loc[df['Survived'] >= 0]);

# Dans data, nous lui avons passé le dataframe avec seulement les données en train. En effet, nous n'avons pas la colonne Survived avec les données de test

# Nous voyons qu'il y a une très grande proportion de femme ( environ 74%) qui ont survécu contrairement au homme (environ 20%)
sns.distplot(df.loc[df['Survived'] == 1, 'Age']);

# On peux voir que les personnes qui ont environ 30 ans avaient plus de 60% de chance de survivre. On peux aussi voir que les enfants ont aussi plus de chance de survie
NUM_FEATURES = ['Age']

CAT_FEATURES = ['Sex']
X_train = df.loc[df['Survived'] >= 0, NUM_FEATURES + CAT_FEATURES]
y_train = df.loc[df['Survived'] >= 0, ['Survived']]
X_test = df.loc[df['Survived'] == -1, NUM_FEATURES + CAT_FEATURES]
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100) # sur la valeur de kaggle, il utilise une ancienne version qui avait par defaut 100 mais depuis qlq mois, scklearn a mis 100 par défault 
rf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

#submission = pd.DataFrame({'PassengerId' : df.loc[df['Survived'] == -1, 'PassengerId'],

#                          'Survived' : rf.predict(X_test)})

#submission.to_csv('submission.csv', index=False)
# Maintenant que notre modèle fonctionne, nous allons essayer de l'améliorer

df.info()
#Essayons de rendre l'apprentissage dynamique

def train_model(rf, NUM_FEATURES, CAT_FEATURES):

    X_train = df.loc[df['Survived'] >= 0, NUM_FEATURES + CAT_FEATURES]

    y_train = df.loc[df['Survived'] >= 0, 'Survived']

    X_test = df.loc[df['Survived'] == -1, NUM_FEATURES + CAT_FEATURES]

    rf.fit(X_train, y_train)

    return {

        'score': accuracy_score(y_train, rf.predict(X_train)),

        'x_test': X_test, 

    }
# Nous pouvons essayer avec d'autre colonne numérique

df['Pclass'].value_counts()
sns.catplot(x="Pclass", y="Survived", kind="bar", data=df.loc[df['Survived'] >= 0]);

# Nous voyons que cette valeur est intéressante car nous voyons que ceux en Pclass 1 ont plus de chance de survie que ceux en 3
NUM_FEATURES.append('Pclass')
# En ajoutant le Parch, on gagne en précision

print(train_model(rf, NUM_FEATURES, CAT_FEATURES)['score'])
# Maintenant que notre modèle fonctionne, nous allons essayer de l'améliorer

df.info()
df.isnull().sum()
# On va regarder du côté de Fare

df['Fare'].value_counts()
sns.catplot(x="Fare", y="Survived", kind="bar", data=df.loc[df['Survived'] >= 0], orient="h");

# En le mettent à la vertical, nous voyons que plus le fare est important, plus les chances de survie sont elevés
# Plus haut nous avons vu qu'une ligne avait une valeur null donc nous la remplacons par la moyenne

df['Fare'].fillna( np.sum(df['Fare'])//len(df.loc[df['Fare'] > 0]) , inplace=True)
# On vérifie

df.isnull().sum()
NUM_FEATURES.append('Fare')
print(train_model(rf, NUM_FEATURES, CAT_FEATURES)['score'])

# Notre score est bien meilleur
# Nous pouvons voir du côté des autres valeurs

df.info();
# Regardons du côté des SibSp

sns.catplot(x="SibSp", y="Survived", kind="bar", data=df.loc[df['Survived'] >= 0], orient="h");

# En le mettent à la vertical, nous voyons que plus le SibSp est important, plus les chances de survie sont faibles
# Essayons avec 

NUM_FEATURES.append('SibSp')

print(train_model(rf, NUM_FEATURES, CAT_FEATURES)['score'])

# Le modèle est un peu plus précis
df.info();
# Le Embarked a l'air intéressant

df['Embarked'].value_counts()
sns.catplot(x="Embarked", y="Survived", kind="bar", data=df.loc[df['Survived'] >= 0]);

# On peut voir que ceux venant du port C avait plus de chance de survie
# Comme vu plus haut, cette colonne contient deux valeurs null alors nous allons les remplacer par le quai le plus utilisés

df['Embarked'].fillna('S', inplace = True)
df['Embarked'] = df['Embarked'].map({

        'S': 0, 

        'C': 1,

        'Q': 3,

}).astype(int)
CAT_FEATURES.append('Embarked')

print(train_model(rf, NUM_FEATURES, CAT_FEATURES)['score'])
# Je pense pas qu'il y a d'autre valeur intéressante, néanmoins, afin d'améliorer la précision, il peut être intéressant de diminuer la plage de valeur

# Commencons par regouper les valeurs de Fare

df['Fare_test'] = pd.qcut(df['Fare'], 10, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8 ,9])

# qcut permet de couper par tranche de n passé en deuxième paramètres, ici c'est 10

# Le labelle permet de passer d'un type objet (catégory) à  des int
df['Fare_test'].describe()
sns.catplot(x="Fare_test", y="Survived", kind="bar", data=df.loc[df['Survived'] >= 0], aspect=3);

# On peux voir que ceux qui ont le plus de fare ont le plus de chance de survie



# Après verification, cette donnée regroupé a fait regresser ma précision
print(train_model(rf, NUM_FEATURES, CAT_FEATURES)['score'])
submission = pd.DataFrame({'PassengerId' : df.loc[df['Survived'] == -1, 'PassengerId'],

                           'Survived' : rf.predict(train_model(rf, NUM_FEATURES, CAT_FEATURES)['x_test'])})

submission.to_csv('submission.csv', index=False)