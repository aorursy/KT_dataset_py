import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
#Chargement des fichier necessaire à la modelisation et affichage

input_path = "../input/titanic/"

df_train = pd.read_csv(input_path+'train.csv')

df_test = pd.read_csv(input_path+'test.csv')

#Affichage

df_train
#Informations sur le type de données

df_train.info()
#Description des données

df_train.describe()
#Verification des valeurs Nan

df_train.isna().sum()
#Moyenne de valeurs ages 

moyenne_age = int(round(df_train['Age'].loc[~df_train['Age'].isna()].mean()))

#Remplacement des valeurs null de l'âge par la moyenne des age

df_train.loc[(df_train['Age'].isna()), 'Age'] = moyenne_age
#Enlèvement les personnes qui n'ont pas embarqués

df_train = df_train.drop(df_train.loc[(df_train['Embarked'].isna()), 'Embarked'].index)

#Enlèvement des colonnes non pertinentes Name & Ticket

df_train = df_train.drop(['Name','Ticket'], axis=1)
#Filtre Survived

df_filtered = df_train.loc[df_train['Survived'] == 1]
# Distribution des Age

sns.distplot(df_filtered.Age);



# Observation

# Le nombre le plus majoré est vers 30 ans

# Le nombre de passager majoritaire est compris entre 15 et 50

# Avec une poignée d'enfants qui ont entre 0 et 10
# Distribution des Sex

sns.countplot(x="Sex", data=df_filtered);



# Observation

# Il y a plus de femmes à bord 
# Comparaison des Sexe et filtrage de l'age

df_filtered = df_filtered.loc[(df_filtered['Age'] >= 15) & (df_filtered['Age'] <= 50)]

sns.countplot(x="Sex", data=df_filtered);



# Observation

# Le nombre de survivants sont surtout des femmes
# Les femmes entre 15 et 45 ans sont surtout dans l'emarquement S

sns.countplot(x="Embarked",data=df_filtered);
## Femme en fonction de leurs Embarquement

df_female = df_filtered.loc[(df_filtered['Sex']=='female') & ((df_filtered['Embarked']=='S') | (df_filtered['Embarked']=='C'))]

sns.countplot(y="Pclass", hue="Sex", data=df_female)
df_female
#Recherche d'un correlation

corr = df_female.corr()

sns.heatmap(corr, annot=True);
NUM_FEATURES = ['Age', 'Pclass', 'Embarked']
X_train = df_female.loc[:, NUM_FEATURES]
y_train = df_train['Survived']
X_test = df_test.loc[:, NUM_FEATURES]
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, y_train)

#Erreur que je n'ai pas compris 
train_acc = accuracy_score(y_train, rf.predict(X_train))

print(train_acc)
rf.predict(X_test)
# cellule pour stocker le résultat du modèle en appelant son modèle rf et son set de test X_test

submission = pd.DataFrame({'PassengerId' : df_test['PassengerId'],

                           'Survived' : rf.predict(X_test)})

submission.to_csv('submission.csv', index=False)