 # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output
# importer les données
df = pd.read_csv('../input/train3.csv',sep = ',')
df[:3]
# représentation graphique
num_years = int(df["Year_of_Release"].max() - df["Year_of_Release"].min() + 1) # nombre d'années considérées
years = df["Year_of_Release"].dropna() # on ne considère pas les cellules vides
years = years.astype(int) 
plt.hist(years, bins=num_years, color="lightskyblue", edgecolor="black") 
plt.title("Distribution du nombre de jeux sortis par année") 
plt.xlabel("Année")
plt.ylabel("Nombre de jeux")
plt.show()
# module avec les modèles de régression
from sklearn import linear_model
# enlever les lignes avec des cellules manquantes
df2 = df.dropna()
# créer les inputs et outputs des modèles
y_na = df2['NA_Sales'].values
y_glob = df2['Global_Sales'].values
X = df2['JP_Sales'].values
# on met les données sous la forme adéquate pour le modèle de régression
X = X.reshape((-1,1))
y_glob = y_glob.reshape((-1,1))
y_na = y_na.reshape((-1,1))
# visualisation des données
plt.scatter(X,y_na)
plt.title('Ventes en Amérique du Nord en fonction des ventes au Japon')
plt.show()
# visualisation des données
plt.scatter(X,y_glob)
plt.title('Ventes totales en fonction des ventes au Japon')
plt.show()
regr_na = linear_model.LinearRegression() #on charge le modèle de régression linéaire
regr_na.fit(X, y_na) #on le fit aux données
# afficher les résultats de la régression
print('RESULTATS POUR LES VENTES EN AMÉRIQUE DU NORD \n')
print ('Pente: ', float(regr_na.coef_))
print ("Ordonnée à l'origine: ", float(regr_eu.intercept_))
print('R^2: ',regr_eu.score(X, y_na))
regr_glob = linear_model.LinearRegression() #on charge le modèle de régression linéaire
regr_glob.fit(X, y_glob) #on le fit aux données
# afficher les résultats de la régression
print('RESULTATS POUR LES VENTES TOTALES \n')
print ('Pente: ', float(regr_glob.coef_))
print ("Ordonnée à l'origine: ", float(regr_glob.intercept_))
print('R^2: ',regr_glob.score(X, y_glob))
dfTest = pd.read_csv('../input/test3.csv',sep = ',')
dfTest[:3]
# créer l'input
Xtest = dfTest['JP_Sales'].values
Xtest = Xtest.reshape((-1,1))
# Ventes en Amerique du Nord
pred_na = regr_na.predict(Xtest)
pred_na
# Ventes totales
pred_glob = regr_glob.predict(Xtest)
pred_glob
# créer le fichier à soumettre
results = pd.DataFrame()
results['ID'] = dfTest['ID']
results['NA_Sales'] = pred_na
results['Global_Sales'] = pred_glob
results[:4]
#results.to_csv('submission_sample.csv',sep = ',',index = False)

