import pandas as pd
""""

from stravalib.client import Client



client = Client(access_token='155f536e6e18aa2a28ca0bca765485261e164290') #updating every 6 hours



activities = client.get_activities()

sample = list(activities)[0]

sample.to_dict()



my_cols =['average_speed',

          'max_speed',

          'average_heartrate',

          'max_heartrate',

          'distance',

          'elapsed_time',

          'moving_time',

          'total_elevation_gain',

          'elev_high',

          'type', 

          'start_date_local',

          'kudos_count']

data = []

for activity in activities:

    my_dict = activity.to_dict()

    data.append([my_dict.get(x) for x in my_cols])

    

df = pd.DataFrame(data, columns=my_cols)

df.to_csv('strava_full_data.csv')

"""
df = pd.read_csv("../input/strava-data/strava_full_data.csv")
df['type'].value_counts()
df.head()
df.shape
df = df[df['type'].isin(['Run','Ride'])]
df.info()
df.drop(['average_heartrate','max_heartrate','Unnamed: 0'], axis=1, inplace=True)
df.info()
df['elev_high'] = df['elev_high'].fillna(value=0)
df.dropna(inplace=True)
df.info()
df['distance'] = df['distance']/1000

df['distance'] = df['distance'].round(2)



df['average_speed'] = 1/(df['average_speed']/100*6)

df['average_speed'] = df['average_speed'].round(2)



df['max_speed'] = 1/(df['max_speed']/100*6)

df['max_speed'] = df['max_speed'].round(2)
df['elapsed_time'].str.len().value_counts()
df[df['elapsed_time'].str.len() == 14]
df.set_value(215,'elapsed_time', df.loc[215,'moving_time'])
df.set_value(559,'elapsed_time', df.loc[559,'moving_time'])
df[df['elapsed_time'].str.len() == 15]
df.set_value(208,'elapsed_time', df.loc[208,'moving_time'])

df.set_value(444,'elapsed_time', df.loc[444,'moving_time'])
def to_minutes(str):

    return sum(i*j for i, j in zip(map(float, str.split(':')), [60, 1, 1/60]))



df['time_min_elapsed'] = df['elapsed_time'].apply(to_minutes)

df['time_min_moving'] = df['moving_time'].apply(to_minutes)



df['time_min_elapsed'] = df['time_min_elapsed'].round(2)

df['time_min_moving'] = df['time_min_moving'].round(2)
df['start_date_local'] = pd.to_datetime(df['start_date_local'])
df
df.groupby('type')['distance'].nlargest(3)
df.groupby('type')['time_min_elapsed'].nlargest(3)
df.loc[1061,:]
data = df[df['type'] == 'Run']
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
%config InlineBackend.figure_format = 'retina'

sns.distplot(data['total_elevation_gain']);
import numpy as np
data.apply(np.max)
features = ['distance', 'total_elevation_gain']

data[features].plot(kind='density', subplots=True, layout=(1, 2), 

                  sharex=False, figsize=(10, 4));
data
data = data.drop(['elapsed_time', 'moving_time','type','kudos_count'], axis=1)
data
data.set_index('start_date_local', inplace=True)

data.head()
data.index
data.isnull().sum()
data.max()
data[data['max_speed'] == data['max_speed'].max()]
data = data.drop(['max_speed'], axis=1)
sns.pairplot(data)
corr = data.corr()

plt.figure(figsize = (12,8))

sns.heatmap(corr, annot=True, fmt=".2f");
trends = data.copy()
trends.set_index(pd.to_datetime(trends.index), drop=True, inplace=True)
trends['weekday'] = trends.index.map(lambda x: x.weekday)
trends
trends.groupby('weekday').mean()
trends.groupby('weekday').mean()['time_min_elapsed'].plot.bar()
import calendar

list(calendar.day_name)



plt.style.use('ggplot')



trends.groupby('weekday').mean()['time_min_elapsed'].plot(kind='bar', figsize=(12,5))

plt.xticks(list(range(7)), list(calendar.day_name), rotation='horizontal')

plt.xlabel('')

plt.ylabel('Time in minutes')

plt.title('Average training time by day of the week')
trends['year'] = trends.index.map(lambda x: x.year)
trends['year'].value_counts()
trends.groupby('year').mean()
trends.groupby('year').mean()['time_min_elapsed'].plot(kind='bar')
data.head()
cols = ['average_speed','distance','total_elevation_gain','time_min_elapsed']

sns.pairplot(x_vars=cols, y_vars=cols, data=data, size=5)
import sklearn

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(data)
data_scaled = pd.DataFrame(X, columns=['average_speed', 'distance', 'total_elevation_gain', 'elev_high', 'time_min_elapsed', 'time_min_moving'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9,8))



ax1.set_title('Before Scaling')

sns.kdeplot(data['average_speed'], ax=ax1)

sns.kdeplot(data['distance'], ax=ax1)

sns.kdeplot(data['total_elevation_gain'], ax=ax1)

sns.kdeplot(data['elev_high'], ax=ax1)

sns.kdeplot(data['time_min_elapsed'], ax=ax1)

sns.kdeplot(data['time_min_moving'], ax=ax1)



ax2.set_title('After Standard Scaler')

sns.kdeplot(data_scaled['average_speed'], ax=ax2)

sns.kdeplot(data_scaled['distance'], ax=ax2)

sns.kdeplot(data_scaled['total_elevation_gain'], ax=ax2)

sns.kdeplot(data_scaled['elev_high'], ax=ax2)

sns.kdeplot(data_scaled['time_min_elapsed'], ax=ax2)

sns.kdeplot(data_scaled['time_min_moving'], ax=ax2)



plt.show()

from sklearn.cluster import KMeans

model = KMeans(n_clusters=5)

model.fit(X)

data['Cluster'] = model.labels_
data['Cluster'].value_counts()
data.groupby('Cluster').mean()
data.groupby('Cluster').std()
Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(X)

    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
data.groupby('Cluster').mean()
data['Cluster'].value_counts()
data[data['Cluster'] == 2]
data[data['Cluster'] == 1]
data[data['Cluster'] == 4]