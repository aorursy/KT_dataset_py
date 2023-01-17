# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd
import matplotlib.pyplot as plt # visualisation
import seaborn as sns # visualisation
import csv
from sklearn.neighbors import KNeighborsClassifier # machine learning
from sklearn.tree import DecisionTreeClassifier # machine learning
from sklearn.naive_bayes import GaussianNB # machine learning
from sklearn.linear_model import Perceptron # machine learning
from sklearn.model_selection import train_test_split # scission train/test
from sklearn.model_selection import KFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df_iris = pd.read_csv("../input/iris.arff.csv")

# Toutes nos variables sont des variables numériques. Il n'y a aucun attribut catégoriel à transformer et aucun attribut binaire.
# Les attributs dont nous disposons sont des variables techniquement continues, mais qui ont été discrétisées sur des plages restreintes. Il n'est donc pas nécessaire de les discrétiser sur des espacements encore plus restreints, au risque de perdre en information.
# Aucune valeur ne manque, comme on peut le voir avec la fonction .info .

#iris.groupby(['class']).size()

# On a exactement autant d'occurences d'éléments pour chacune des classes : notre jeu de données est équilibré.

foret = []
naives = []
cerveau = []


for i in range(250):
    df_iris = df_iris.sample(frac=1)

    df_test = df_iris.iloc[0:50]
    df_train = df_iris.iloc[50:150]

    df_train_input = df_train.drop("class", axis=1)
    df_train_class = df_train["class"]
    df_test_input = df_test.drop("class", axis=1)
    df_test_class = df_test["class"]

    #Arbre de décision
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(df_train_input, df_train_class)
    acc_decision_tree = round(decision_tree.score(df_test_input, df_test_class) * 100, 2)
    foret.append(acc_decision_tree)

    #Classification naîve Bayésienne
    gaussian = GaussianNB()
    gaussian.fit(df_train_input, df_train_class)
    acc_gaussian = round(gaussian.score(df_test_input, df_test_class) * 100, 2)
    naives.append(acc_gaussian)

    #Réseau de neurones
    perceptron = Perceptron()
    perceptron.fit(df_train_input, df_train_class)
    acc_perceptron = round(perceptron.score(df_test_input, df_test_class) * 100, 2)
    cerveau.append(acc_perceptron)

print("Précision moyenne arbre de décision : ",np.mean(foret))
print("Précision moyenne naîve bayes : ",np.mean(naives)) 
print("Précision moyenne réseau de neurones : ",np.mean(cerveau))

#On constate que la classification par arbre de décision ou naîve Bayes sont très efficaces.
#Toutefois, le perceptron donne un résultat bien en-deçà des performances des deux autres classifieurs.
#Le perceptron est très efficace pour catégoriser des objets dont les classes sont séparables (schématiquement) par une ligne droite.
#La non-précision du perceptron nous permet de théoriser que la distribution des classes au sein du dataset iris ne suit pas une telle distribution.
#Le bon fonctionnement de l'arbre de décision nous permet de déterminer que notre jeu de données n'est pas sujet à de subtiles perturbations : il est très propre et permet d'obtenir un classifieur fiable.
#Naive Bayes nous donne une haute précision, ce qui confirme notre postulat de départ de ne pas re-discrétiser nos attributs sur des plages plus restreintes.

#Pour affiner la précision de ces tests, on décide de faire varier les tailles de nos sets training/test. Pour éviter l'overfitting, on isole 10 rows de notre jeu de données total.
#Ainsi, on a un jeu effectif de 140 rows. On décide donc d'essayer diverses tailles de sets training/test, en partant de training 135 test 5, et en incrémentant de 5 en 5.
#On plottera alors la précision de notre modèle par arbre de décision (choisi arbitrairement) suivant cette variation de longueurs de sets, pour décider du meilleur choix.

foret = []

for i in range(1,28):
    k = 5*i
    df_val = df_iris.iloc[140:150]
    df_test = df_iris.iloc[0:k]
    df_train = df_iris.iloc[k:140]
    
    df_test = pd.concat([df_test,df_val])

    df_train_input = df_train.drop("class", axis=1)
    df_train_class = df_train["class"]
    df_test_input = df_test.drop("class", axis=1)
    df_test_class = df_test["class"]

    #Arbre de décision
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(df_train_input, df_train_class)
    acc_decision_tree = round(decision_tree.score(df_test_input, df_test_class) * 100, 2)
    foret.append(acc_decision_tree)
    
plt.plot(foret)

#On constate que notre choix initial de séparation 50/100 donne une des meilleures précisions tout en évitant l'overfitting ou l'underfitting : c'était donc un choix pertinent qu'on peut conserver.
