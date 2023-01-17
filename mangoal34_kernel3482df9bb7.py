import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import warnings



warnings.simplefilter("ignore")

%matplotlib inline

print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_submission = pd.read_csv('../input/sample_submission.csv')
df.info()
df.head()
df.describe().drop("count")
sns.boxplot(x=df['trip_duration']);

#On peut observer que des données ne peuvent etre prise en compte, il va faloir filtrer 
import math



def ft_haversine_distance(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371 #km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h
def distance (df):

    df['distance'] = ft_haversine_distance(df['pickup_latitude'].values,

                                                 df['pickup_longitude'].values, 

                                                 df['dropoff_latitude'].values,

                                                 df['dropoff_longitude'].values)

    return df
#fonction recuperation des dates



def rescu_date (df):

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])



    df['year'] = df['pickup_datetime'].dt.year

    df['month'] = df['pickup_datetime'].dt.month

    df['day'] = df['pickup_datetime'].dt.day

    df['hour'] = df['pickup_datetime'].dt.hour

    df['weekday'] = df['pickup_datetime'].dt.weekday

    df['minute'] = df['pickup_datetime'].dt.minute

    



    return df
#Ajout de la colonne distance

df = distance(df)

#Ajout des columns date 

rescu_date(df);
#Suite au check des datas, on procede au filtre du dataset



#Filtre les lignes avec moins d'une heure de trajet 

df = df[(df.trip_duration < 3600)]



#On récupere les coordonnées cohérente 

df = df[(df.dropoff_latitude < 41)]

df = df[(df.dropoff_latitude > 40)]



df = df[(df.dropoff_longitude > -79)]

df = df[(df.dropoff_longitude < -72)]



df = df[(df.pickup_latitude < 51)]

df = df[(df.passenger_count > 0)]



#On ne prend pas en compte les distance de trajet nul

df = df[(df.distance > 0)]
df.head()
df.describe().drop("count") #Permet d'afficher les valeurs non scientifiques
#On selectionne les colonnes à récuperer 

X_column_features = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'year', 'month', 'day', 'hour', 'weekday', 

                      'distance', 'minute'];



#On selectionne la target 

y_column_target = ['trip_duration'];



#Fonction qui permet de splitter les collones 

def split_dataset(df):

    X = df[X_column_features]

    y = df[y_column_target]

    return X, y
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (mean_squared_error, SCORERS)

from sklearn.model_selection import cross_val_score

import math



rf = RandomForestRegressor()
X_train, y_train = split_dataset (df)
#On créé des dataset contenant les données pour le train et on check leurs contenue 

X_train, y_train = split_dataset (df)

X_train.shape, y_train.shape
from sklearn.model_selection import ShuffleSplit

#On split les données pour amélioré la vitesse d'éxecution 

rs = ShuffleSplit(n_splits=3, train_size=.25, test_size=.6)
#On lance un cross validation avec une output

math.sqrt((-cross_val_score(rf, X_train, y_train, cv=rs, scoring='neg_mean_squared_log_error', n_jobs=-1)).mean())
rf.fit(X_train, y_train)
#On ajoute les données dans la data de test

X_test = rescu_date(df_test)

X_test = distance(X_test)

X_test = df_test[X_column_features];



#On verifie son contenu 

X_test.head()
#On lance la prédiction

y_pred = rf.predict(X_test)
#On envoi les résultats de la prediction dans le fichier csv à envoyer pour le submit

df_submission['trip_duration'] = y_pred

df_submission.head()
df_submission.to_csv('submission.csv', index=False)
!ls #On verifie la présence du fichier