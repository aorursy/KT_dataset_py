#Importation des librairies



import os



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



%matplotlib inline
# Load datas



TRAIN_PATH = os.path.join('..', 'input', 'train.csv')



df = pd.read_csv(TRAIN_PATH, index_col=0)

df.head()
# Affichage des informations sur le dataset



df.info()
# Affichage des valeurs de répartition du dataset



df.describe()
# Recherche de valeurs dupliquées



df.duplicated().sum()
# Suppression des enregistrements dupliqués et vérification du nombre total



df_no_duplicates = df.drop_duplicates()

df_no_duplicates.shape[0]
# Recherche des valeurs manquantes



df_no_duplicates.isna().sum()
# Recherche d'éventuels outliers



sns.boxplot(x=df_no_duplicates['trip_duration']).set_title("Boxplot de la durée des trajets")

plt.show();
# Affichage des 4 outliers dont la durée du trajet dépasse 100000



df_no_duplicates.loc[df_no_duplicates['trip_duration'] > 100000]
# Suppression des outliers



df_clean = df_no_duplicates.loc[df_no_duplicates['trip_duration'] < 100000]

df_clean.shape[0]
# Création de features de distance entre le départ et l'arrivée pour la longitude et la latitude



df_enhanced = df_clean.copy()



df_enhanced["delta_longitude"] = df_clean["pickup_longitude"] - df_clean["dropoff_longitude"]

df_enhanced["delta_latitude"] = df_clean["pickup_latitude"] - df_clean["dropoff_latitude"]

df_enhanced["delta_total"] = np.sqrt(np.square(df_enhanced["delta_longitude"]) + np.square(df_enhanced["delta_latitude"]))
# Création de features à partir des informations du pickup_datetime



df_enhanced["pickup_Timestamp"] =  pd.to_datetime(df_clean["pickup_datetime"], format='%Y/%m/%d')

df_enhanced["pickup_hour"] = df_enhanced["pickup_Timestamp"].dt.strftime('%-H').astype(int)

df_enhanced["pickup_minute"] = df_enhanced["pickup_Timestamp"].dt.strftime('%-M').astype(float)

df_enhanced["pickup_daynumber"] = df_enhanced["pickup_Timestamp"].dt.strftime('%w').astype(int)

df_enhanced["pickup_month"] = df_enhanced["pickup_Timestamp"].dt.strftime('%m').astype(int)

df_enhanced["pickup_weeknumber"] = df_enhanced["pickup_Timestamp"].dt.strftime('%U').astype(int)
# Selection des features



df_features = df_enhanced[['vendor_id', 'passenger_count', 'pickup_hour', 'pickup_minute', 'pickup_daynumber', 'pickup_weeknumber', 'pickup_month', 'delta_longitude', 'delta_latitude', 'delta_total']]



df_target = np.log(df_enhanced['trip_duration'].values)

df_features.head()
# Découpage du dataset en données d'entrainement et en données de test



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.1, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Utilisation du modèle RandomForestRegressor



from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=10, min_samples_split=30, max_depth=20, random_state=42, n_jobs=-1)
# Entrainement du modèle RandomForestRegressor



rf.fit(X_train, y_train)
# Vérification du RMSE de ce modèle



from sklearn.metrics import mean_squared_error

from math import sqrt



rms = sqrt(mean_squared_error(y_test, rf.predict(X_test)))
rms
# Chargement du dataset de soumission



TEST_PATH = os.path.join('..', 'input', 'test.csv')

test = pd.read_csv(TEST_PATH, index_col=0)
# Calculs des nouvelles features sur ce nouveau dataset



test["delta_longitude"] = test["pickup_longitude"] - test["dropoff_longitude"]

test["delta_latitude"] = test["pickup_latitude"] - test["dropoff_latitude"]

test["delta_total"] = np.sqrt(np.square(test["delta_longitude"]) + np.square(test["delta_latitude"]))



test["pickup_Timestamp"] =  pd.to_datetime(test["pickup_datetime"], format='%Y/%m/%d')

test["pickup_hour"] = test["pickup_Timestamp"].dt.strftime('%-H').astype(int)

test["pickup_minute"] = test["pickup_Timestamp"].dt.strftime('%-M').astype(float)

test["pickup_daynumber"] = test["pickup_Timestamp"].dt.strftime('%w').astype(int)

test["pickup_weeknumber"] = test["pickup_Timestamp"].dt.strftime('%U').astype(int)

test["pickup_month"] = test["pickup_Timestamp"].dt.strftime('%m').astype(int)
# Prédictions de la durée des trajets du dataframe de test.



test_features = test[['vendor_id', 'passenger_count', 'pickup_hour', 'pickup_minute', 'pickup_daynumber', 'pickup_weeknumber', 'pickup_month', 'delta_longitude', 'delta_latitude', 'delta_total']]



y_pred = np.exp(rf.predict(test_features))

# Préparation de la soumission



submission = pd.DataFrame({'id': test.index, 'trip_duration': y_pred})



submission.to_csv('submission.csv', index=False)


