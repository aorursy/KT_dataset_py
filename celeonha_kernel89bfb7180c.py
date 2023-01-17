#importation des paquets pour l'analyse des données
import numpy as np
import pandas as pd
import random as rnd
#importation des paquets pour la visualisation
import matplotlib.pyplot as plt
import seaborn as sns
#importation des paquets pour accéder aux algorithmes de machine learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
#Chargement des données
df_trainingSet = pd.read_csv('../input/iris.arff.csv')
df_trainingSet.head()
df_trainingSet.tail()
df_trainingSet.info()
df_trainingSet.describe()
df_trainingSet.describe(include=['O'])
df_trainingSet.describe(include=['O'])
#Nous voulons maintenant savoir quels attributs est correlé à la classe c'est-à-dire influence la longueur du sépale
#Le cas du sépale
df_trainingSet[['class', 'sepallength']].groupby(['class'], as_index=False).mean().sort_values(by='sepallength', ascending=False)
#Nous voulons maintenant savoir quels attributs est correlé à la classe c'est-à-dire influence la largeur du sépale
df_trainingSet[['class', 'sepalwidth']].groupby(['class'], as_index=False).mean().sort_values(by='sepalwidth', ascending=False)
#Nous voulons maintenant savoir quels attributs est correlé à la classe c'est-à-dire influence la longueur du pétale
#Le cas du pétal
df_trainingSet[['class', 'petallength']].groupby(['class'], as_index=False).mean().sort_values(by='petallength', ascending=False)
#Nous voulons maintenant savoir quels attributs est correlé à la classe c'est-à-dire influence la largeur du pétale
df_trainingSet[['class', 'petalwidth']].groupby(['class'], as_index=False).mean().sort_values(by='petalwidth', ascending=False)
g = sns.FacetGrid(df_trainingSet, col='class')
g.map(plt.hist, 'sepallength', bins=10)
#On constate que dans le cas des Iris-setosa, la longueur des sépales est souvent 5