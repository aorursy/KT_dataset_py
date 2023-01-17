# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import seaborn as sb

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/california-housing-prices/housing.csv")
#Afficher les details des colonnes

data.info()
#récupérer toutes les colonnes non numériquee

text_col= data.select_dtypes(include=['object']).columns

#calculer nombre de catégorie pour chaque colonne

#convertir le type de colonnes au type category

for col in text_col :

    print (col + ":", len (data[col].unique()))

    data [col]= data[col].astype('category')
#calculer le nombre d'observation pour chaque catégorie

data['ocean_proximity'].value_counts()
#gérer les caractéristiques factices

for col in text_col:

    col_dummines= pd.get_dummies(data[col], prefix="proximity")

    data =pd.concat([data,col_dummines],axis=1)

    del data[col] #supprimer la colonne d'origine
#la somme des variables null de chaque colonne de dataframe

data.isnull().sum()
#supprimer les observations qui contiennent des valeurs null

data=data.dropna(axis=0)
#la somme des variables null de chaque colonne de dataframe

data.isnull().sum()
#afficher les 10 premières lignes de dataset

data.head(10)
#fractionner les données en deux groupes test et train mélanger

train,test = train_test_split(data, test_size=0.25, random_state=42)
#calculer la moyenne , ecart_type , min et max pour chaque colonnes

train.describe()
#fractionner les variables descriptives et la variable cible

x_train = train.drop(['median_house_value'],axis=1)

y_train = train.median_house_value

x_test= test.drop('median_house_value', axis=1)

y_test = test.median_house_value
#matrice de corrélation

caracteristique = data.columns.tolist() #récupérer la liste des caractéristiques

matrice_correlation = data[caracteristique].corr() #calculer la corrélation entre les caractéristiques

corr_median_house_value= matrice_correlation['median_house_value'].abs().sort_values() 

corr_median_house_value
#heatmap

correlation_sup = corr_median_house_value[corr_median_house_value > 0.1]#récupérer les corrélations supérieures à 0.1

matrice_correlatation_supp= data[correlation_sup.index].corr().abs() 

sb.heatmap(matrice_correlatation_supp) #afficher le Heatmap
#afficher l'histogramme pour chaque colonne

x_train.hist(bins=50, figsize=(20,20))

plt.show()
#StandardScaler()

min_max_scaler = StandardScaler()



x_train_sco= min_max_scaler.fit_transform(x_train)

x_test_sco= min_max_scaler.fit_transform(x_test)

x_train = pd.DataFrame(x_train_sco, columns=x_train.columns)

x_test = pd.DataFrame(x_test_sco, columns=x_test.columns)

x_train.head(10)
#entrainer le modèle

model = LinearRegression()



model.fit(x_train[['median_income','proximity_INLAND','proximity_<1H OCEAN','proximity_NEAR BAY','latitude','total_rooms','housing_median_age','households','total_bedrooms','longitude','population']],y_train)
#calculer les prédictions de test_set

prediction = model.predict(x_test[['median_income','proximity_INLAND','proximity_<1H OCEAN','proximity_NEAR BAY','latitude','total_rooms','housing_median_age','households','total_bedrooms','longitude','population']])
#comparer les résultats avec les valeurs observer

resultat = pd.DataFrame({'Predicted':prediction,'Actual':test['median_house_value']})

resultat
from sklearn.metrics import mean_squared_error

#calculer RMSE

mse = mean_squared_error(resultat['Actual'], resultat['Predicted'])

rmse = np.sqrt(mse)

rmse
#calculer le score

model.score(x_train[['median_income','proximity_INLAND','proximity_<1H OCEAN','proximity_NEAR BAY','latitude','total_rooms','housing_median_age','households','total_bedrooms','longitude','population']],y_train)
resultat= resultat.reset_index()

resultat= resultat.drop(['index'],axis=1)

plt.plot(resultat[:100])

plt.legend(['Actual','Predicted'])
from sklearn.ensemble import RandomForestRegressor

modelRandomForest = RandomForestRegressor()

modelRandomForest.fit(x_train,y_train)

modelRandomForest.score(x_train,y_train)
output = modelRandomForest.predict(x_test)

modelRandomForest.score(x_test,test.median_house_value)

output_csv = pd.DataFrame({'Label':output})



output_csv.to_csv('output.csv',index=False)