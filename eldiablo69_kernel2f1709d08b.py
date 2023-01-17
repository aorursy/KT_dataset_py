

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans

from matplotlib import pyplot

from mpl_toolkits.mplot3d import Axes3D

# Input data files are available in the "../input/" directory.

import os

DataFrame=pd.read_csv('../input/data.csv')

# On restreint la base de données aux 800 joueurs les plus chers pour ne pas avoir à traiter 

# des cas trop différents

DataFrame=DataFrame[0:800]

# Any results you write to the current directory are saved as output.



#On commence par créer un filtre sur le potentiel et l'âge 

# de manière à conserver les joueurs qui risquent de percer

JeunesPepites=DataFrame[(DataFrame['Potential']>92) & (DataFrame['Age']<26)]

print(JeunesPepites['Name'])



# On peut ensuite regarder la valeur des joueurs 

# Il faut donc convertir leur valeur (chaine alphanumérique) en nombre

Value=DataFrame['Value'].map(lambda x: x.lstrip('€').rstrip('M'))

Value=pd.to_numeric(Value)



###



#On va maintenant réaliser notre premier clustering

# L'objectif est de repérer les joueurs avec une faible valeur, un gros potentiel et un petit âge

# On commence par classifier selon 2 variables : potentiel et valeur

Potential=DataFrame['Potential']

Age=DataFrame['Age']

Value=DataFrame['Value'].map(lambda x: x.lstrip('€').rstrip('M'))

#pyplot.scatter(Potential,Value)

PetroGreen=list(zip(Potential,Value))

PetroGreen=pd.DataFrame(PetroGreen)

# On utilise 3 clusters afin de représenter les bons coups en bleu (gros potentiel et pas cher)

# les mauvais coups en violot (cher et faible potentiel) et les autres en jaunes

kmeans=KMeans(n_clusters=3, random_state=0).fit(PetroGreen)

y_kmeans=kmeans.predict(PetroGreen)

#pyplot.scatter(PetroGreen.iloc[:,0],PetroGreen.iloc[:,1],c=y_kmeans,s=50, cmap='viridis')



###



# On classifie maintenant selon 3 variables : potentiel, valeur et âge

Potential=DataFrame['Potential']

Age=DataFrame['Age']

Value=DataFrame['Value'].map(lambda x: x.lstrip('€').rstrip('M'))

#pyplot.scatter(Potential,Value,Age)

PetroGreen=list(zip(Potential,Value,Age))

PetroGreen=pd.DataFrame(PetroGreen)

# On utilise 3 clusters afin de représenter les bons coups en bleu (gros potentiel et pas cher)

# les mauvais coups en violet (cher et faible potentiel) et les autres en jaunes

kmeans=KMeans(n_clusters=3, random_state=0).fit(PetroGreen)

y_kmeans=kmeans.predict(PetroGreen)

fig=pyplot.figure()

ax=fig.add_subplot(111,projection='3d')

pyplot.scatter(PetroGreen.iloc[:,0],PetroGreen.iloc[:,1],PetroGreen.iloc[:,2],c=y_kmeans)

#On observe que le nuage n'est pas le même que précedemment

#Cela semble logique car l'âge a une importance pour le recruteur

#Un joueur vieux mais ayant un bon potentiel et une faible valeur sera classé par le recruteur comme

#un bon coup dans le premier (donc en bleu)

#mais comme un coup moyen dans ce second cas (donc en jaune)