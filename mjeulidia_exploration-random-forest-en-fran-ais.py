import pandas as pd  

import numpy as np

import pandas_profiling

import seaborn as sns

import geopandas as gpd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
#df = pd.read_csv("C:/Users/Marie-JeanneVieille/Documents/Happiness_2017.csv")

df = pd.read_csv("../input/world-happiness/2017.csv")

print("Le fichier a " + str(df.shape[0]) + " lignes et " + str(df.shape[1]) + " colonnes")
#Liste des colonnes et leur type 

df.dtypes
# 5 premières lignes du dataset

df.head(5)
pandas_profiling.ProfileReport(df)
# Matrice des corrélations : 

cor = df.corr() 

sns.heatmap(cor, square = True, cmap="coolwarm",linewidths=.5,annot=True )

#Pour choisr la couleur du heatmap : https://matplotlib.org/examples/color/colormaps_reference.html
#Chargement du fonds de carte 

# Dispo ici https://tapiquen-sig.jimdofree.com/english-version/free-downloads/world/

map_df = gpd.read_file('../input/world-country/World_Countries.shp')



#Jointure avec nos données (on ne conserve que Country et Happiness.Rank)

map_df = map_df.set_index('COUNTRY').join(df[['Country','Happiness.Score']].set_index('Country'))

map_df.dropna(inplace=True)

map_df.reset_index(inplace=True)



#Préparation de la carte

# on fixe les seuils pour la couleur

vmin, vmax = 0, 8

# création de la figure et des axes

fig, ax = plt.subplots(1, figsize=(18, 5))



# Création de la carte

map_df.plot(column='Happiness.Score', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')

# On supprime l'axe des abscisses

ax.axis('off')



# On ajoute un titre

ax.set_title('Happiness.Score par pays', fontdict={'fontsize': '16', 'fontweight' : '2'})



# On créé la légende

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []



# On ajoute la légende

cbar = fig.colorbar(sm)
# On transforme Country en index

pd.DataFrame.set_index(df, 'Country',inplace=True)



# On supprime 3 colonnes

df.drop(columns =['Happiness.Rank','Whisker.high', 'Whisker.low' ], inplace=True)

#On stocke Happiness.Score (la variable à prédire) dans cible

cible = np.array(df['Happiness.Score'])



#On supprime Happiness.Score du dataset

df= df.drop('Happiness.Score', axis = 1)



#On conserve les noms de variable à part

liste_variables = list(df.columns)



#On convertit le dataset en array

df = np.array(df)
#On créé 4 dataset : 

#   - x_train contient 75% de x  

#   - y_train contient le appiness.Score associé à x_train

# => x_train et y_train permettront d'entraîner l'algorithme

#

#   - x_test contient 25% de x  

#   - y_test contient le appiness.Score associé à x_test

# => x_test et y_test permettront d'évaluer la performance de l'algorithme une fois entrainé sur le train



x_train,x_test,y_train,y_test=train_test_split(df,cible,test_size=0.25, random_state=2020)
#On importe l'algorithme à partir de sklearn

from sklearn.ensemble import RandomForestRegressor



#On créé un Random Forest de 100 arbres 

rf = RandomForestRegressor(n_estimators = 100, random_state = 2020)



#Et on lance le training sur notre dataset de train

rf.fit(x_train, y_train)
#On applique le modèle que l'on vient d'entraîner sur l'échantillon de test

predictions = rf.predict(x_test)
#On va calculer plusieurs erreurs entre la valeur prédite et le score de bonheur réel (que nous avions stocké dans y_test)

#     - MAE : Mean Asolute Error

#     - MAPE : Mean Absolute Percentage Error 



# MAE 

erreurs = abs(predictions - y_test)

print('Mean Absolute Error:', round(np.mean(erreurs), 2))
# MAPE

mape = 100 * (erreurs / y_test)

print('Mean Absolute Percentage Error :', round(np.mean(mape), 2), '%.')
importances = rf.feature_importances_

indices = np.argsort(importances)



# style du graphique 

plt.style.use('fivethirtyeight')

%matplotlib inline



plt.figure(1)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [liste_variables[i] for i in indices])

plt.xlabel('Relative Importance')