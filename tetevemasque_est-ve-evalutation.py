import pandas as pd

import seaborn as sns

import pathlib as Path

import matplotlib.pyplot as plt

import sklearn

import numpy as np

import math

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit



import os



%matplotlib inline



print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv', index_col=0)

df.head()
durations_n = df[(df.store_and_fwd_flag == 'N')]['trip_duration']

durations_y = df[(df.store_and_fwd_flag == 'Y')]['trip_duration']



durations_y.head()
plt.boxplot([durations_n, durations_y])
plt.boxplot([durations_n])
# Filtrer les durées trop longues

durations_n = durations_n[(durations_n < 2000)]

durations_n = durations_n[(durations_n > 60)]
plt.boxplot([durations_n])
plt.boxplot([durations_y])
# Filtrer les durées trop longues

durations_y = durations_y[(durations_y < 5000)]

durations_y = durations_y[(durations_y > 60)]
plt.boxplot([durations_y])
passagers = df.groupby('passenger_count')['passenger_count']

passagers.describe()
# Create data



g1 = (df['pickup_longitude'], df['pickup_latitude'])

g2 = (df['dropoff_longitude'], df['dropoff_latitude'])



data = (g1, g2)

colors = ("red", "blue")

groups = ("Pickup", "Dropoff")



# Create plot

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, facecolor="1.0")



for data, color, group in zip(data, colors, groups):

    x, y = data

    ax.scatter(x, y, alpha=0.5, c=color, edgecolors='none', s=30, label=group)



plt.title('Matplot scatter plot')

plt.legend(loc=2)

plt.show()

# Ajoute des colonnes de date au dataframe passé en paramètre

def date_cols(df) :

    

    # Convertir les colonnes de date au format 'datetime'

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])



    # Ajouter des colonnes pour chaque partie des dates (pickup, dropoff)

    # (penser a faire une fonction ensuite)



    df['pickup_year'] = df['pickup_datetime'].dt.year

    df['pickup_month'] = df['pickup_datetime'].dt.month

    df['pickup_day'] = df['pickup_datetime'].dt.day

    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    df['pickup_minutes'] = (df['pickup_hour'] * 60) + df['pickup_datetime'].dt.minute

    df['pickup_seconds'] = (df['pickup_minutes'] * 60) + df['pickup_datetime'].dt.second

    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday



date_cols(df)
# Fonction qui traite la colonne du flag (remplace Y par 1, N par 0)

def flag_col(df) :

    

    booleanDictionary = {'Y': 1, 'N': 0}

    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map(booleanDictionary)



flag_col(df)
def ft_haversine_distance(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371 # Rayon de la Terre (Km)

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



df['distance'] = ft_haversine_distance(df['pickup_latitude'], df['pickup_longitude'], df['dropoff_latitude'], df['dropoff_longitude'])
# Vérifier qu'il n'y a pas de valeurs récurrentes

df.duplicated().sum()
# Supprimer les valeurs récurrentes

df = df.drop_duplicates()

df.duplicated().sum() # On revérifie...
# Vérifier qu'il n'y a pas de valeurs nulles

df.isna().sum()
df.describe()
df = df[(df.trip_duration < 5000)]
# Colonnes à inclure

selected_columns = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',

                    'dropoff_latitude', 'pickup_month' , 'pickup_hour','pickup_minutes',

                    'pickup_seconds', 'pickup_weekday', 'pickup_day', 'distance']



# X_train = données, y_train = colonne à prédire

X_train = df[selected_columns]

y_train = df['trip_duration']
X_train.head()
rf = RandomForestRegressor()



# On va splitter les données à tester pour rendre le test moins long

sp = ShuffleSplit(n_splits=3, train_size=.25, test_size=.12)



score = -cross_val_score(rf, X_train, y_train, cv=sp, scoring='neg_mean_squared_log_error')

score = [np.sqrt(l) for l in score]

score[:5]
rf.fit(X_train, y_train)
df_test = pd.read_csv('../input/test.csv', index_col=0)

df_test.head()
# traitement pour les colonnes de date

date_cols(df_test)
# traitement pour la colonne de flag

flag_col(df_test)
# traitement pour la distance

df_test['distance'] = ft_haversine_distance(df_test['pickup_latitude'], df_test['pickup_longitude'], df_test['dropoff_latitude'], df_test['dropoff_longitude'])
X_test = df_test[selected_columns]

X_test.head()
y_pred = rf.predict(X_test)

y_pred.mean()
X_test.index.shape, y_pred.shape
submission = pd.read_csv('../input/sample_submission.csv') 

submission.head()
submission['trip_duration'] = y_pred

submission.head()
submission.describe()
submission.to_csv('submission.csv', index=False)