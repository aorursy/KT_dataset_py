# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from haversine import haversine

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

from sklearn.model_selection import train_test_split, cross_val_score

import warnings

warnings.filterwarnings('ignore')







print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importation du fichier CSV (en indiquant que la colonne IDENTITY est la colonne id du dataset)

data = pd.read_csv('../input/train.csv') #, index_col = 0)
test = pd.read_csv('../input/test.csv') #, index_col = 0)
test.shape
test.head()
data.head()
data.info
test.shape
# Convertir les dates de timestamp en datetime afin d'extraire d'autres détails importants de la date

data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])



# Convertir les dates de timestamp en datetime afin d'extraire d'autres détails importants de la date

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
# Extraction, Calcul et affectation des nouvelles données relative à la pickup_date dans le dataset

data['weekday'] = data.pickup_datetime.dt.weekday_name

data['month'] = data.pickup_datetime.dt.month

data['weekday_num'] = data.pickup_datetime.dt.weekday

data['pickup_hour'] = data.pickup_datetime.dt.hour



# Extraction, Calcul et affectation des nouvelles données relative à la pickup_date dans le dataset

test['weekday'] = test.pickup_datetime.dt.weekday_name

test['month'] = test.pickup_datetime.dt.month

test['weekday_num'] = test.pickup_datetime.dt.weekday

test['pickup_hour'] = test.pickup_datetime.dt.hour
# Fonction de calcul de distance entre les points de départs et les points d'arrivées

# Elle prend en paramètre le dataset, et renvoie un vecteur contenant les distances entre ces points

# Elle applique la méthode de Haversine pour le calcul des distances entre deux coordonnées

def calcul_distance(df):

    pickedup = (df['pickup_latitude'], df['pickup_longitude'])

    dropoff = (df['dropoff_latitude'], df['dropoff_longitude'])

    return haversine(pickedup, dropoff)
# Calcul des distances entre les points de départs et les points d'arrivées

# et les mettant dans une nouvelle colonne distance

data['distance'] = data.apply(lambda x : calcul_distance(x), axis = 1)
test['distance'] = test.apply(lambda x : calcul_distance(x), axis = 1)
data.dtypes.reset_index()
test.dtypes.reset_index()
# Découper les features catégoriques en plusieurs variables numériques / indicatrices



dummy = pd.get_dummies(data.store_and_fwd_flag, prefix='flag')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

data = pd.concat([data,dummy], axis = 1)



dummy = pd.get_dummies(data.vendor_id, prefix='vendor_id')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

data = pd.concat([data,dummy], axis = 1)



dummy = pd.get_dummies(data.month, prefix='month')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

data = pd.concat([data,dummy], axis = 1)



dummy = pd.get_dummies(data.weekday_num, prefix='weekday_num')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

data = pd.concat([data,dummy], axis = 1)



dummy = pd.get_dummies(data.pickup_hour, prefix='pickup_hour')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

data = pd.concat([data,dummy], axis = 1)



dummy = pd.get_dummies(data.passenger_count, prefix='passenger_count')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

data = pd.concat([data,dummy], axis = 1)
test.head()
# Découper les features catégoriques en plusieurs variables numériques / indicatrices



dummy = pd.get_dummies(test.store_and_fwd_flag, prefix='flag')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

test = pd.concat([test,dummy], axis = 1)



dummy = pd.get_dummies(test.vendor_id, prefix='vendor_id')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

test = pd.concat([test,dummy], axis = 1)



dummy = pd.get_dummies(test.month, prefix='month')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

test = pd.concat([test,dummy], axis = 1)



dummy = pd.get_dummies(test.weekday_num, prefix='weekday_num')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

test = pd.concat([test,dummy], axis = 1)



dummy = pd.get_dummies(test.pickup_hour, prefix='pickup_hour')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

test = pd.concat([test,dummy], axis = 1)



dummy = pd.get_dummies(test.passenger_count, prefix='passenger_count')

dummy.drop(dummy.columns[0], axis=1, inplace=True) #enlever la première colonne qui est l'index

test = pd.concat([test,dummy], axis = 1)
test.shape
data.head()
data.shape
test.head()
pd.options.display.float_format = '{:.2f}'.format #Basculer l'affichage des floats en format scientifique
data.passenger_count.value_counts()
test.passenger_count.value_counts()
print(data.passenger_count.describe())

print(f'median = {data.passenger_count.median()}')

# On remarque que la moyenne, la médiane et les modes sont presque égaux à 1
# Alors on remplace le passenger_count 0 par 1

data['passenger_count'] = data.passenger_count.map(lambda x: 1 if x == 0 else x)
test['passenger_count'] = test.passenger_count.map(lambda x: 1 if x == 0 else x)
test.shape
data.passenger_count.value_counts()
test.passenger_count.value_counts()
#Nombre de courses par nombre de passagers

sns.countplot(data.passenger_count)

plt.show()
data.dtypes.reset_index()
test.dtypes.reset_index()
# Distribution des horaires de départs des courses sur 24 heures

sns.countplot(data.pickup_hour)

plt.show()
data.head()
#Vérifiez d'abord l'index des features et le label

list(zip( range(0,len(data.columns)),data.columns))
SELECTED_COLUMNS = ['vendor_id', 'vendor_id_2', 

                    'flag_Y', 

                    'pickup_hour', 'distance', 

                    'month','weekday_num',

                    'month_2', 'month_3', 'month_4', 'month_5', 'month_6',

                    'weekday_num_1', 'weekday_num_2', 'weekday_num_3', 'weekday_num_4', 'weekday_num_5', 'weekday_num_6',

                    'passenger_count_1', 'passenger_count_2', 'passenger_count_3', 'passenger_count_4', 'passenger_count_5', 'passenger_count_6',

                    'pickup_hour', 'pickup_hour_1', 'pickup_hour_2','pickup_hour_3', 'pickup_hour_4', 'pickup_hour_5','pickup_hour_6',  'pickup_hour_7', 'pickup_hour_8', 

                    'pickup_hour_9', 'pickup_hour_10', 'pickup_hour_11', 'pickup_hour_12', 'pickup_hour_13', 'pickup_hour_14', 'pickup_hour_15', 'pickup_hour_16', 

                    'pickup_hour_17', 'pickup_hour_18', 'pickup_hour_19', 'pickup_hour_20', 'pickup_hour_21', 'pickup_hour_22', 'pickup_hour_23' ]

X_many_features = data[SELECTED_COLUMNS]

X_many_features.head()

y_many_features = np.log1p(data['trip_duration'])

X_many_features.shape, y_many_features.shape
rf = RandomForestRegressor(random_state=42)

rf.fit(X_many_features, y_many_features)
cv_scores = -cross_val_score(rf, X_many_features, y_many_features, cv=3, scoring='neg_mean_squared_error')

cv_scores
cv_scores.mean()
X_test = test[SELECTED_COLUMNS]

predictions = np.exp(rf.predict(X_test))-np.ones(len(X_test))



X_test.shape

pred = pd.DataFrame(predictions, index=test['id'])

pred.columns = ['trip_duration']

pred.to_csv("submission_.csv")



pd.read_csv('submission_.csv')