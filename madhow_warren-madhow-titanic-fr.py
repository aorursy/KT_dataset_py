# Cet environnement en Python 3 est accompagné de plusieurs librairies d'analyses utiles installées

# Il est défini par l'image docker de kaggle/python : https://github.com/kaggle/docker-python



# Chargement des librairies :



import numpy as np  # algèbre linéaire

import pandas as pd # traitement de données

import sklearn # machine learning

import seaborn as sns # visualisation de data basé sur matplotlib

import matplotlib.pyplot as plt



# affichage avec la bibliothèque graphique intégrée à Notebook

%matplotlib inline



# non affichage des messages de prévention

import warnings

warnings.filterwarnings('ignore')



# répartition du dataset dans train et test

from sklearn.model_selection import train_test_split



# algorithmes de machine learning qui combinent de nombreux modèles d'apprentissage pour créer un modèle prédictif solide

from sklearn.ensemble import GradientBoostingClassifier



# précision du score de classification

from sklearn.metrics import accuracy_score
# chargement des datasets de train et test



input_path = "../input/titanic/"



train = pd.read_csv(input_path + 'train.csv')

test = pd.read_csv(input_path + 'test.csv')
# affichage de des premieres lignes du train



train.head()
# affichage de toutes les colonnes



train.columns
test.head()
# suppression des colonnes non nécessaires dans nos 2 datasets



train.drop(['Cabin'],axis=1,inplace=True)

train.drop(['Ticket'],axis=1,inplace=True)

train.drop(['PassengerId'],axis=1,inplace=True)

train.drop(['Name'],axis=1,inplace=True)

train.drop(['Fare'],axis=1,inplace=True)



test.drop(['Cabin'],axis=1,inplace=True)

test.drop(['Ticket'],axis=1,inplace=True)

test.drop(['PassengerId'],axis=1,inplace=True)

test.drop(['Name'],axis=1,inplace=True)

test.drop(['Fare'],axis=1,inplace=True)
test.head()
train.head()
# visualisation graphique des survivants



sns.countplot(x = 'Survived', hue = 'Sex', data = train)
# visualisation des survivants ayant entre 0 et plusieurs parents/enfants



sns.countplot(x = 'Survived', hue = 'Parch', data = train )
# visualisation des survivants ayant entre 0 et plusieurs frères ou sœurs/épou﹒x﹒se﹒s



sns.countplot(x = 'Survived', hue = 'SibSp', data = train)
# visualisation des survivants par classe



sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
# verification de l'existence de variables nulles



train.isnull().sum()
test.isnull().sum()
# calcul de la moyenne d'age des personnes



train['Age'].mean()
# remplacement des valeurs nulles par la moyenne dans la colonne 'Age'



train['Age'].fillna((train['Age'].mean()), inplace = True)

test['Age'].fillna((test['Age'].mean()), inplace = True)
# verification des valeurs nulles si toujours existantes

test.isnull().sum()
train.head()
# transformation de certaines variables de train en dummies (factices)



train_Pclass = pd.get_dummies(train['Pclass'], drop_first = True)

train_Sex = pd.get_dummies(train['Sex'], drop_first = True)

train_Embarked = pd.get_dummies(train['Embarked'], drop_first = True)
# transformation de certaines variables de test en dummies



test_Pclass = pd.get_dummies(test['Pclass'], drop_first = True)

test_Sex = pd.get_dummies(test['Sex'], drop_first = True)

test_Embarked = pd.get_dummies(test['Embarked'], drop_first = True)
# ajout des colonnes des données factices 



train = pd.concat([train, train_Pclass, train_Sex, train_Embarked], axis = 1)

test = pd.concat([test, test_Pclass, test_Sex, test_Embarked], axis = 1)
# suppression des anciennes colonnes



train.drop(['Sex','Embarked','Pclass'], axis = 1, inplace = True)

test.drop(['Sex','Embarked','Pclass'], axis = 1, inplace = True)
train.head()
test.head()
# test modele



y = train['Survived']
y.head()
X = train.drop('Survived', axis = 1)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)
gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)
y_predict = gbc.predict(X_test)
round(accuracy_score(y_predict, y_test) * 100, 2)
# chargement du fichier d'exemple de soumission pour copier la forme du tableau pour le fichier de soumission

sub = pd.read_csv(input_path + 'gender_submission.csv')
sub
# creation de notre fichier de soumission à partir de l'exemple en changeant les valeurs de 'Survived' avec nos valeurs prédites 



predictions = gbc.predict(test)

sub['Survived'] = predictions

sub.to_csv('submit.csv', index = False)

sub