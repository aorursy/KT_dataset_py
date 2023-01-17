import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")
%matplotlib inline
# We start by reading the data
df = pd.read_csv('../input/autotel-shared-car-locations/sample_table.csv')
df.sample(5)
df.describe()
df = df[df['total_cars'] > 0]
df_cars_by_time = df.groupby('timestamp').agg({'total_cars':'sum'}).reset_index()
df_cars_by_time.sample(5)
df_cars_by_time['timestamp'] = df_cars_by_time['timestamp'].apply(pd.Timestamp)
df_cars_by_time.set_index('timestamp').sort_index().rolling('60min').mean().plot(figsize=(20,6), c='salmon', lw=1.6)
plt.grid()
plt.show()
df_cars_by_time['usage_rate'] = (260 - df_cars_by_time['total_cars']) / 260
df_cars_by_time.set_index('timestamp').sort_index()['usage_rate'].rolling('60min').mean().plot(figsize=(20,6), c='mediumslateblue', lw=1.6)
plt.grid()
plt.show()
df_cars_by_time['usage_rate'] = (260 - df_cars_by_time['total_cars']) / 260
df_cars_by_time.set_index('timestamp').sort_index()['usage_rate'].rolling('3D').mean().plot(figsize=(20,6), c='navy', lw=1.6)
plt.grid()
plt.show()
# Convert timezone
timestamps = pd.DatetimeIndex(df_cars_by_time['timestamp'])
timestamps = timestamps.tz_convert('Asia/Jerusalem')

df_cars_by_time['local_time'] = timestamps

#Extract time features
df_cars_by_time['weekday'] = df_cars_by_time['local_time'].dt.weekday_name
df_cars_by_time['hour'] = df_cars_by_time['local_time'].dt.hour
df_cars_by_time.head() # Looks right!
plt.figure(figsize=(20,6))
plt.subplot(121)
sns.barplot(x='hour', y='total_cars', data=df_cars_by_time)
plt.subplot(122)
sns.boxplot(x='weekday', y='total_cars', data=df_cars_by_time, showfliers=False)

plt.show()
df_cars_by_time['usage_rate'] = (260 - df_cars_by_time['total_cars']) / 260
plt.figure(figsize=(20,6))
plt.subplot(121)
sns.barplot(x='hour', y='usage_rate', data=df_cars_by_time)
plt.title('Cars usage rate by hour of day')
plt.subplot(122)
sns.boxplot(x='weekday', y='usage_rate', data=df_cars_by_time, showfliers=False)
plt.title('Cars usage rate by day of week')

plt.show()
import folium
from folium.plugins import HeatMap
df_locations = df.groupby(['latitude', 'longitude', 'timestamp']).sum().reset_index().sample(1500)
df_locations.head()
m = folium.Map([df_locations.latitude.mean(), df_locations.longitude.mean()], zoom_start=11)
for index, row in df_locations.iterrows():
    folium.CircleMarker([row['latitude'], row['longitude']],
                        radius=row['total_cars'] * 6,
                        fill_color="#3db7e4", 
                       ).add_to(m)
    
points = df_locations[['latitude', 'longitude']].as_matrix()
m.add_children(HeatMap(points, radius=15)) # plot heatmap

m
from shapely.geometry import Point, Polygon
from shapely import wkt
df_neighborhood = pd.read_csv('../input/tel-aviv-neighborhood-polygons/tel_aviv_neighborhood.csv')
df_neighborhood.head()
def load_and_close_polygon(wkt_text):
    poly = wkt.loads(wkt_text)
    point_list = poly.exterior.coords[:]
    point_list.append(point_list[0])
    
    return Polygon(point_list)
# Lets transform the WKS's to Polygon Objects and save it to a GeoPandas DataFrame
df_neighborhood['polygon'] = df_neighborhood['area_polygon'].apply(load_and_close_polygon)
neighborhood_map = df_neighborhood.set_index('neighborhood_name')['polygon'].to_dict()
sample_df = df.sample(10000)
sample_df['points'] = sample_df.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1)
sample_df.head()
poly_idxs = sample_df['points'].apply(lambda point : np.argmax([point.within(polygon) for polygon in list(neighborhood_map.values())]))
poly_idxs = poly_idxs.apply(lambda x: list(neighborhood_map.keys())[x])
sample_df['neighborhood'] = poly_idxs.values
sample_df.head()
plt.figure(figsize=(20,7))
sns.barplot(x='neighborhood', y='total_cars', data=sample_df.groupby('neighborhood').count().reset_index())
plt.xticks(rotation=45)
plt.show()
import lightgbm as lgb
df_sample = df.copy()
df_timestamps = pd.DataFrame()
df_timestamps['timestamp'] = df_sample.timestamp.drop_duplicates()
timestamps = pd.DatetimeIndex(df_timestamps['timestamp']).tz_localize('UTC')
df_timestamps['local_time'] = timestamps.tz_convert('Asia/Jerusalem')
df_sample = df_sample.merge(df_timestamps, on='timestamp', how='left')
# Again no reason to calculate on duplocate points, it's very expensive!
df_points = df_sample[['longitude','latitude']].drop_duplicates()
df_points['points'] = df_points.apply(lambda row : Point([row['longitude'], row['latitude']]), axis=1)
poly_idxs = df_points['points'].apply(lambda point : np.argmax([point.within(polygon) for polygon in list(neighborhood_map.values())]))
poly_idxs = poly_idxs.apply(lambda x: list(neighborhood_map.keys())[x])
df_points['neighborhood'] = poly_idxs.values
df_sample = df_sample.merge(df_points[['longitude', 'latitude', 'neighborhood']], on=['longitude', 'latitude'], how='left')
df_sample['time_in_seconds'] = pd.to_datetime(df_sample['local_time']).values.astype(np.int64) // 10**6

seconds_in_day = 24 * 60 * 60
seconds_in_week = 7 * seconds_in_day

df_sample['sin_time_day'] = np.sin(2*np.pi*df_sample['time_in_seconds']/seconds_in_day)
df_sample['cos_time_day'] = np.cos(2*np.pi*df_sample['time_in_seconds']/seconds_in_day)

df_sample['sin_time_week'] = np.sin(2*np.pi*df_sample['time_in_seconds']/seconds_in_week)
df_sample['cos_time_week'] = np.cos(2*np.pi*df_sample['time_in_seconds']/seconds_in_week)

df_sample['weekday'] = df_sample['local_time'].dt.weekday
df_sample['hour'] = df_sample['local_time'].dt.hour

df_sample.sample(5)
aggs = {}
aggs['total_cars'] = 'sum'
aggs['sin_time_day'] = 'mean'
aggs['cos_time_day'] = 'mean'
aggs['sin_time_week'] = 'mean'
aggs['cos_time_week'] = 'mean'
aggs['weekday'] = 'first'
aggs['hour'] = 'first'
df_sample = df_sample.set_index('local_time').groupby([pd.Grouper(freq='60s'), 'neighborhood']).agg(aggs).reset_index()
df_sample.sample(6)
df_sample['neighborhood'] = df_sample['neighborhood'].astype('category')
df_sample['weekday'] = df_sample['weekday'].astype('category')
df_sample['hour'] = df_sample['hour'].astype('category')
df_train = df_sample[df_sample['local_time'] < '2019-01-04']
df_test = df_sample[df_sample['local_time'] >= '2019-01-04']

print('train_shape: ', df_train.shape)
print('test_shape: ', df_test.shape)
features = ['neighborhood', 'sin_time_day', 'cos_time_day', 'sin_time_week', 'cos_time_week', 'weekday', 'hour']
target = 'total_cars'
gbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=250)
gbm.fit(df_train[features], df_train[target],
        eval_set=[(df_test[features], df_test[target])],
        eval_metric='mse',
        early_stopping_rounds=5,
      )
df_test['prediction'] = gbm.predict(df_test[features])
df_test.plot(kind='scatter', x='total_cars', y='prediction', lw=0, s=0.4, figsize=(20,6))
plt.show()