import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from subprocess import check_output

import os

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head()
df_train.describe()

#df_train.info()
df_test.head()
df_test.describe()

#df_test.info()
plt.subplots(figsize=(18,6))

plt.title("outliers")

df_train.boxplot();
#il y a quatre valeurs aberrantes dans le tableau. Cela peut influencer le test alors on utilise cette formule pour réduire l'écart

df_train = df_train[(df_train.trip_duration < 10000) & (df_train.trip_duration > 100)]
#on supprime id et store_and_fwd_flag

df_train.drop(["id"], axis=1, inplace=True)

df_train.drop(["store_and_fwd_flag"], axis=1, inplace=True)

df_test.drop(["store_and_fwd_flag"], axis=1, inplace=True)
df_train["pickup_datetime"] = pd.to_datetime(df_train.pickup_datetime)

df_test["pickup_datetime"] = pd.to_datetime(df_test.pickup_datetime)

df_train.info()
df_train['week'] = df_train.pickup_datetime.dt.week

df_train['weekday'] = df_train.pickup_datetime.dt.weekday

df_train['hour'] = df_train.pickup_datetime.dt.hour

df_train.drop(['pickup_datetime'], axis=1, inplace=True)

df_train.drop(['dropoff_datetime'], axis=1, inplace=True)

df_test['week'] = df_test.pickup_datetime.dt.week

df_test['weekday'] = df_test.pickup_datetime.dt.weekday

df_test['hour'] = df_test.pickup_datetime.dt.hour

df_test.drop(['pickup_datetime'], axis=1, inplace=True)

pass_count = df_train['passenger_count']

print("Maximum number of passengers on a trip : ", np.max(pass_count.values))

print("Minimum number of passengers on a trip : ", np.min(pass_count.values))

print("Average number of passengers on a trip : ", np.mean(pass_count.values))



f = plt.figure(figsize=(10,5))

pass_count = df_train['passenger_count'].value_counts()

sns.barplot(pass_count.index, pass_count.values, alpha=0.7)

plt.xlabel('Number of passengers on a trip', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.show()
#Le nombre de collectes est-il identique pour tout le mois? Découvrons-le

f = plt.figure(figsize=(15,5))

sns.countplot(x='week', data=df_train)

plt.xlabel('Day of month', fontsize=14)

plt.ylabel('Pickup count', fontsize=14)

plt.show()
#les trois jours plus fournir sont le jeudi vendredi et samedi 

f = plt.figure(figsize=(15,5))

days = [i for i in range(7)]

sns.countplot(x='weekday', data=df_train)

plt.xlabel('Day of the week', fontsize=14)

plt.ylabel('Pickup count', fontsize=14)

plt.xticks(days, ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))

plt.show()


f = plt.figure(figsize=(15,5))

sns.countplot(x='hour', data=df_train)

plt.xlabel('Hour', fontsize=14)

plt.ylabel('Pickup count', fontsize=14)

plt.show()
#on peut déduire que l'heure et le jour influence sur la durée total d'un trajet. 
#on va utiliser trip duration pour faire notre prediction.tout engardant en tête que d'autre parametre peuvent influencer notre prédiction voir eda

y = df_train["trip_duration"]

df_train.drop(["trip_duration"], axis=1, inplace=True)

X = df_train

X.shape, y.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
from sklearn.ensemble import RandomForestRegressor
m1 = RandomForestRegressor(n_estimators=19, min_samples_split=2, min_samples_leaf=4, max_features='auto', max_depth=80, bootstrap=True)

m1.fit(X_train, y_train)

m1.score(X_valid, y_valid)

test_columns = X_train.columns

predictions = m1.predict(df_test[test_columns])
my_submission = pd.DataFrame({'id': df_test.id, 'trip_duration': predictions})

my_submission.head()
my_submission.to_csv("submit_file.csv", index=False)