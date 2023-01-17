import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



ROWS_SAMPLE = 8000



df = pd.read_csv("../input/train.csv")



# Delete locations far away from the others

longitude_limit = [-74.027, -73.85]

latitude_limit = [40.67, 40.85]

df = df[(df.pickup_longitude.between(longitude_limit[0], longitude_limit[1], inclusive=False))]

df = df[(df.dropoff_longitude.between(longitude_limit[0], longitude_limit[1], inclusive=False))]

df = df[(df.pickup_latitude.between(latitude_limit[0], latitude_limit[1], inclusive=False))]

df = df[(df.dropoff_latitude.between(latitude_limit[0], latitude_limit[1], inclusive=False))]



df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])



df_trimmed = df.sample(ROWS_SAMPLE)

df_trimmed.head()
longitude = list(df_trimmed.pickup_longitude) + list(df_trimmed.dropoff_longitude)

latitude = list(df_trimmed.pickup_latitude) + list(df_trimmed.dropoff_latitude)



data = pd.DataFrame({'latitude': latitude, 'longitude': longitude})

sns.set_style("white")

ax = sns.regplot(x="longitude", y="latitude", data=data, scatter=True, fit_reg=False, scatter_kws={"s": 0.3})

plt.show()
ax = sns.countplot(df.pickup_datetime.dt.day)

ax.set(xlabel='Days', ylabel='Trips')

plt.show()
ax = sns.countplot(df.pickup_datetime.dt.month)

ax.set(xlabel='Months', ylabel='Trips', xticklabels=['Jan', 'Feb', 'Mar', 'Abr', 'May', 'June'])

plt.show()
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=20, random_state=0).fit(data)

data['kmean_label'] = kmeans.labels_



plt.figure(figsize=(10, 10))

for label in data.kmean_label.unique():

    label_data = pd.DataFrame({'latitude': data.latitude[data.kmean_label == label], 'longitude': data.longitude[data.kmean_label == label]})

    sns.regplot(x="longitude", y="latitude", data=label_data, scatter=True, fit_reg=False, scatter_kws={"s": 2})



plt.show()
df.log_trip_duration = np.log(df.trip_duration)

ax = sns.distplot(df.log_trip_duration, bins=100, kde=False)

plt.xlabel('Trip duration')

plt.ylabel('Frequency')

plt.show()
from haversine import haversine

df_trimmed['trip_distance'] = [haversine((row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude'])) for index, row in df_trimmed.iterrows()]



# Remove trip with high duration

MAX_TRIP_DURATION = 2000

df_trimmed_duration = df_trimmed[(df_trimmed.trip_duration < MAX_TRIP_DURATION)]



ax = sns.regplot(x="trip_duration", y="trip_distance", data=df_trimmed_duration, scatter=True, fit_reg=False, scatter_kws={"s": 1})

ax.set(xlabel='Trip duration', ylabel='Trip distance')

plt.show()