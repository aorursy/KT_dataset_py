import os



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



%matplotlib inline
TRAIN_FILEPATH = os.path.join('..', 'input', 'train.csv')

TEST_FILEPATH = os.path.join('..', 'input', 'test.csv')



df_train = pd.read_csv(TRAIN_FILEPATH)

df_test = pd.read_csv(TEST_FILEPATH)



df_train.head()
# Affichage du nombre de trajets en fonction du temps

df_train["trip_duration"].hist(bins=100)
# Zoom sur les 7000 premières secondes

df_train.loc[df_train["trip_duration"] < 7000, "trip_duration"].hist(bins=100)
# Mise en exergue des outliers en utilisant le logarithme

df_train['trip_duration'].hist(bins=100, log=True);
# Durée moyenne d'un trajet (15 - 16 minutes)

df_train['trip_duration'].mean() / 60
# On retrouve des trajets de moins d'une minute pouvant fausser le calcul de durée

df_train.loc[df_train['trip_duration'] < 300, 'trip_duration'].hist(bins=150)
# Suppression des trajets de moins de 2 minutes et trajets de plus d'1h30

df_train = df_train[df_train['trip_duration'] > 50] # Représente 2% du dataset

df_train = df_train[df_train['trip_duration'] < 3500]
# Affichage du nombre de données manquantes (is NaN)

df_train.isna().sum()
# Affichage du nombre de données nulles

df_train.isnull().sum()
# Suppression des lignes en double en filtrant par l'id

df_train_dupdropped = df_train.drop_duplicates(subset='id')



# Regarde combien de lignes ont étés supprimées

len(df_train) - len(df_train_dupdropped)
# Gestion des données catégoriques

CAT_COL = "store_and_fwd_flag"

df_train[CAT_COL] = df_train[CAT_COL].astype('category').cat.codes

df_test[CAT_COL] = df_test[CAT_COL].astype('category').cat.codes
import math

from geopy.distance import geodesic



def compute_distance(df):

    pointA = (df["pickup_latitude"], df["pickup_longitude"])

    pointB = (df["dropoff_latitude"], df["dropoff_longitude"])



    return geodesic(pointA, pointB).miles
# Calcul des distances sur le dataset train

df_train["distance"] = df_train.apply(compute_distance, axis=1)

df_train.head(20)
# Calcul des distances également sur le dataset test

df_test["distance"] = df_test.apply(compute_distance, axis=1)

df_test.head(20)
# Nombre de trajets à faible distance

df_train[df_train['distance'] < 0.3]['distance'].hist(bins=100)
# Suppression des trajets à faible distance

df_train = df_train[df_train['distance'] > 0.3]
# Calcul des vitesses moyennes des taxis

speed = df_train['distance'] / ( df_train['trip_duration'] / 3600 )

df_train['speed'] = speed



df_train.head(20)
# Nombre de taxis roulant à plus de 75mph

df_train[speed > 75].count()
# Affichage de ces taxis

df_train[speed > 75].head(40)
# Suppression de ces taxis

df_train = df_train[speed < 75]
df_train["pickup_hour"] = pd.to_datetime(df_train.pickup_datetime).dt.hour

df_train["pickup_day"] = pd.to_datetime(df_train.pickup_datetime).dt.dayofweek



df_test["pickup_hour"] = pd.to_datetime(df_test.pickup_datetime).dt.hour

df_test["pickup_day"] = pd.to_datetime(df_test.pickup_datetime).dt.dayofweek



df_train.head()
# Nombre de trajets par heure

df_train["pickup_hour"].plot.hist(bins=24, title="Fréquence des trajets par heure")
# Fréquence des trajets par jour

df_train['pickup_day'].plot.hist(bins=7)
df_train.isnull().sum()
df_test.isna().sum()
FEATURES = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude", "pickup_hour", "pickup_day", "distance"]

TARGET = 'trip_duration'
X_train = df_train[FEATURES]

Y_train = df_train[TARGET]



X_test = df_test[FEATURES]



X_train.shape, Y_train.shape
from sklearn.ensemble import RandomForestRegressor



m1 = RandomForestRegressor(n_estimators=15)

m1.fit(X_train, Y_train)
from sklearn.model_selection import cross_val_score



cv_scores = cross_val_score(m1, X_train, Y_train, cv=5, scoring='neg_mean_squared_log_error')

cv_scores
predictions = m1.predict(X_test)

predictions[:10]
submission = pd.DataFrame({'id': df_test['id'], 'trip_duration': predictions})

submission.to_csv('submission.csv', index=False)



submission.head()