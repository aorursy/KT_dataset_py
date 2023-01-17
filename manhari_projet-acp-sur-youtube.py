# Librairies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Récupération de la base de données
df = pd.read_csv("../input/FRvideos.csv")

# Élimination des lignes indésirables
df = df[df.comments_disabled == False] # On souhaite les commentaires activés
df = df[df.ratings_disabled == False] # On souhaite les likes et dislikes activés
df = df[df.video_error_or_removed == False] # On souhaite la vidéo active à l'époque du dataset
df = df.dropna() # On souhaite des données complètes

# Stockage du nombre d'individus
rows = df.shape[0]

# Premier affichage
df.head()
# On conserve les 4 colonnes qui nous intéressent
select = ['views', 'likes', 'dislikes', 'comment_count']

# On normalise la dataset ainsi conservée
dfnorm = df[select]
dfnorm = StandardScaler().fit_transform(dfnorm)

# On vérifie que la normalisation est un succès
print("Moyennes :\n", np.mean(dfnorm, axis = 0)) # Les moyennes devraient valoir 0 ou être infiniment petites avec l'arrondi
print("\nÉcarts-types :\n", np.std(dfnorm, axis = 0)) # Les écarts-types devraient valoir 1
print("\nMatrice de corrélation :\n", (1/rows) * np.matmul(np.transpose(dfnorm), dfnorm)) # La matrice doit avoir des 1 sur la diagonale
# ACP bénigne afin d'utiliser ses fonctions de calculus de valeurs propres et variances
acp = PCA()
acp.fit_transform(dfnorm)

# Éléments nécessaire au choix du nombre de composantes principales
print("Valeurs propres :\n", acp.explained_variance_)
print("\nQualité de représentation des axes en % :\n", acp.explained_variance_ratio_ * 100)
print("\nQualité cumulée en % :\n", np.cumsum(acp.explained_variance_ratio_) * 100)
# On génère ici l'ACP sur laquelle nous allons travailler
n = 2 # Nombre de composantes principales
acp = PCA(n)
composantes = acp.fit_transform(dfnorm)

# On affiche les premiers résultats
print("Valeurs propres :\n", acp.explained_variance_) # On rappelle les valeurs propres conservées
print("\nMatrice de changement de base :\n", acp.components_) # La matrice de changement de base, obtenue normalement à partir des vecteurs propres
print("\nComposantes principales :\n", composantes) # Les composantes principales données par la matrice de changement de base
print("\nSomme des composantes :\n", composantes.sum(axis = 0)) # La somme par colonne des composantes principales doit valoir 0
# On observe les représentations des individus sur chaque axe
quality = composantes ** 2
for i in range(n):
   quality[:,i] = quality[:,i] / np.sum(dfnorm ** 2, axis=1)
print("Pourcentage de représentation des individus par axe :\n", quality * 100)
# On observe les contributions des individus aux axes
contrib = composantes ** 2
for i in range(n):
    contrib[:,i] = contrib[:,i] / (rows * acp.explained_variance_[i])

print("Contribution aux axes :\n", contrib)
# Création des figures
fig, axes = plt.subplots(figsize=(12, 12))

# Sélectionner les 10 premiers éléments comme exemple
tmp = pd.DataFrame(columns=list(df))
for i in range(10):
    tmp.loc[i]=df.iloc[i]
    plt.annotate(tmp.loc[i].channel_title, (composantes[i, 0], composantes[i, 1]))
    
# Échelle des axes
plt.plot([-2,2], [0,0])
plt.plot([0,0], [-2,2])

# Affichage du graphique et du dataset
plt.show()
select2 = ['channel_title', 'views', 'likes', 'dislikes', 'comment_count']
tmp[select2]
# Corrélations variables-facteurs
x = np.sqrt(acp.explained_variance_)
y = np.zeros((len(select), len(select)))
for i in range(n):
    y[:,i] = acp.components_[i,:] * x[i]

print(pd.DataFrame({'Variable':select, 'Composante 1':y[:,0], 'Composante 2':y[:,1]}))
# Création des figures
fig, axes = plt.subplots(figsize=(12,12))

# Récupération des noms
for i in range(len(select)):
    plt.annotate(select[i],(y[i,0], y[i,1]))

# Échelle des axes
plt.plot([-1,1],[0,0])
plt.plot([0,0],[-1,1])

# Ajouter un cercle
cercle = plt.Circle((0,0), 1, fill=False)
axes.add_artist(cercle)

# Affichage du cercle
plt.show()