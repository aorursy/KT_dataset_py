# data preparation

# data exploration



import numpy as np # linear algebra

import pandas as pd # manipulation de tableaux de données (dataframes)



# colonne = label ou feature

# ligne = entrée du dataframe



import json

import seaborn as sns 

import matplotlib.pyplot as plt # visualisation de données

%matplotlib inline # affichage des graphiques dans le notebook



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
# affiche la version de pandas

print(pd.__version__)
# réglage des options pandas (facultatif)

pd.options.display.max_rows = 100
# affiche les fichiers data

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# importation de la data par pandas

train = pd.read_csv("../input/LA_TRANSITION_ECOLOGIQUE.csv")
# affiche le type de la data

print(type(train))
# affiche les 5 premières lignes du tableau data

train.head()
# affiche les 5 dernières lignes

train.tail()
# affiche la dimension (nombre de lignes, nombre de colonnes)

train.shape
# vérification des lignes doublons

train.drop_duplicates() 

train.shape
# affiche les infos

train.info()
# affiche les colonnes

print(train.columns)
# affiche le type des colonnes

print(train.dtypes)
# affiche la description

print(train.describe(include='all'))
# affiche le champ titre

print(train['title'])
# affiche un autre champ

print(train['createdAt'])
# affiche un autre champ

print(train['reference'])
# affiche de nouveau le champ titre

print(train.title)
# affiche plusieurs champs de colonnes

print(train[['authorId','authorType']])
# affiche les 5 premières lignes de la colonne sélectionnée

print(train['title'].head())
# idem

print(train.title.head())
# affiche des stats

print(train['title'].describe())
# affiche un calcul de valeurs

print(train['title'].value_counts())
# affiche le titre index 3

print(train['title'][3])
# affiche les titres de l'index 0 à 3

print(train['title'][0:4])
# affiche les titres de l'index 2 à 4 (index 5 moins 1)

print(train['title'][2:5])
train.head()
train['authorType'].head(10)
train_drop = train.drop(['id', 'reference', 'createdAt', 'publishedAt', 'updatedAt', 'trashed', 'trashedStatus', 'authorId'], axis =  1)
train = train_drop
train.head()
train.sort_values(by = "authorZipCode")
# affiche les 5 premières lignes (index 0 à 4) et les deux premières colonnes

print(train.iloc[0:5,0:2])
# affiche les trois premières lignes (index 0 à 2) et les colonnes 0, 2 et 6

print(train.iloc[0:3,[0,2,6]])
train.head(10)