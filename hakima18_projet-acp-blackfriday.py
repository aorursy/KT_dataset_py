# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Chargement des données
base = pd.read_csv('../input/BlackFriday.csv')

n = 1000 #Variable utiliser pour définire le nombre d'individuts
base2 = base.head(n) #Echantillon de n individuts
#Supprimer les colonnes 
base3 = base2.drop(labels=['Product_ID', 'Age', 'City_Category', 'Stay_In_Current_City_Years', 'Occupation', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'User_ID'], axis = 1)
var = base2['Product_ID'] #Stocké la colonne Product_ID dans une variable

#Remplacer le F par 1 et le M par 0    
for i in range(0,n) : 
    if base3.Gender[i] == "F":
        base3.Gender[i] = "1"
    if base3.Gender[i] == "M":
        base3.Gender[i] = "0"

#Affichage de la base apres le pré-traitement
print(base3)
#Dimension
print(base3.shape)
#Nombre d'individus
n = base3.shape[0]
#Nombre de variable
p = base3.shape[1]
#Première étape déterminer la matrice centrée réduite 
import sklearn
from sklearn.preprocessing import StandardScaler
#Instanciation de la classe 
sc = StandardScaler()
#XS matrice centrée réduite 
XS = sc.fit_transform(base3)
print("Matrice centrée réduite : \n", XS)
#Vérification : matrice de dimension n * p
print("La matrice est de dimension : ", XS.shape)
#Vérification : somme colonnes = 0
print("La somme des colonnes : \n", sum(XS))
#Vérification : variance colonnes = 1
print("La variance des colonnes : \n",np.var(XS, axis=0))
#Deuxième étape matrice de corrélation
XSt = np.transpose(XS)
corr = (1/n)*np.matmul(XSt,XS)
print("Matrice de corrélation : \n", corr)
#Vérification matrice d'ordre p
print("La matrice de corrélation est d'ordre : ", corr.shape)
#Vérification matrice symétrique 
#Fonction supplémentaire 
def check_symmetric(a, tol=1e-8):
    return not False in (np.abs(a-a.T) < tol)
print("La matrice de corrélation est symétrique : ", check_symmetric(corr))
#Vérification Diagonal de 1
print("La diagonale est égale à : \n", np.diag(corr))
#Vérification coefficients compris -1 et 1
print("Les coefficients sont différent de 1 et -1 : ", (corr > 1).all() and (corr < -1).all())
#Troisieme étape composante principale
from sklearn.decomposition import PCA
#instanciation
acp = PCA()
#calculs des coordonnées factorielles
coord = acp.fit_transform(XS)
print("Composantes principales : \n", coord)
#nombre de composantes calculées
print("Nombre de composante K : ", acp.n_components_)
#Vérification dimension n * p
print("Matrice de dimension n x p : ",coord.shape)
#Vérification somme colonnes = 0
print('Matrice centrée : somme colonnes = 0 \n', coord.sum(axis = 0))
#Vérification décroissantes
#Quatrieme étape valeur/vecteur propres
#Les valeus propres
vp = acp.explained_variance_
print("Les valeurs propres : \n", vp)
#Qualité de représentation
print("\nLa qualité de représentation : \n", (acp.explained_variance_ratio_*100), "%")
#Les vecteur propres
print("\nMatrice de changement de base : \n",acp.components_)
#Qualité de représentation de chaque individu
#Contribution des individus dans l'inertie totale
d = np.sum(XS**2, axis = 1)
cos = coord**2
for j in range(p):
    cos[:,j] = cos[:,j]/d
print(pd.DataFrame({'id' : base3.index, 'Qlt axe 1' : cos[:,0], 'Qlt axe 2' : cos[:,1], 'Qlt axe 3' : cos[:,2]}))
#print(np.sum(cos, axis=1))
#Cinquième étape matrice des saturations
#racine carrée des valeurs propres 
vp2 = np.sqrt(vp)
#corrélation des variables avec les axes
corvar = np.zeros((p,p))
for k in range(acp.n_components_):
    corvar[:,k] = acp.components_[k,:] * vp2[k]
print("Matrice de saturation \n", corvar)
#Vérification carrée d'ordre p
print(corvar.shape)
#Vérification coefficients entre -1 et 1
#Sixième étape représentation graphiques
#Cercle des corrélations
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)

for j in range(p):
    plt.annotate(base3.columns[j],(corvar[j,0],corvar[j,1]))
    
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)

cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
plt.show
#Représentation des individuts dans le plan 
fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(-6,6)
axes.set_ylim(-6,6)

for i in range(100):
    plt.annotate(var.index[i], (coord[i,0],coord[i,1]))

plt.plot([-6,6],[0,0], color='silver', linestyle='-',linewidth=1)
plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)

plt.show()
base3.head(70)