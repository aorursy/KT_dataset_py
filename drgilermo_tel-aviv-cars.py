import numpy as np 
import pandas as pd 
import tqdm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import geopy.distance
import folium
import folium.plugins as plugins
import seaborn as sns
from sklearn.cluster import KMeans

print(os.listdir("../input"))
df = pd.read_csv(r'../input/autotel-shared-car-locations//sample_table.csv')
df['carsList'] = df.carsList.apply(lambda x: x[1:-1]) # remove square brackets
df['carsList'] = df.carsList.apply(lambda x: x.split(',')) # convert string to list
df['carsList'] = df.carsList.apply(lambda x: [] if x == [''] else x) # denote empty lists
df['carsList'] = df.carsList.apply(lambda x: [int(i) for i in x]) # convert list items to int
df['total_cars'] = df.carsList.apply(len) 
df = df[df.total_cars > 0]
# Parse list of cars into different rows 
def explode(df, lst_cols, fill_value=''):
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]
    
new_df = explode(df, ['carsList'], fill_value='')
# Pivot the table to a new structures, where the indices are unique timestamps, the columns are cars and the values are the coordinates of the cars
pivot_df = new_df.pivot(index='timestamp',columns='carsList', values=['latitude', 'longitude'])
pivot_df.head()
def get_car_trips(pivot_df, car_num):
    # First, take the relevant columns for the car in question
    car = pivot_df[[('latitude', car_num), ('longitude', car_num)]]
    car = car[pd.isnull(car[('latitude', car_num)]) == False]
    
    # Find the previous location
    car.loc[:, 'prev_lat'] = car.shift()[('latitude', car_num)]
    car.loc[:, 'prev_lon'] = car.shift()[('longitude', car_num)]
    
    # If the location has not changed, there is no trip going on
    car.loc[:, 'trip'] = car[('latitude', car_num)] == car.prev_lat
    car.loc[:, 'trip'] = car.trip.apply(lambda x: 0 if x else 1)
    car.loc[:, 'trip'] = car.trip.cumsum()
    car.reset_index(inplace=True)
    
    # Merge the data frame with itself shifted by one
    f = {'timestamp': ['min', 'max'], ('latitude', car_num): 'first', ('longitude', car_num): 'first'}
    trip_df = car.groupby('trip').agg(f)
    prev_df = car.groupby('trip').agg(f).shift()

    trip_df = pd.merge(trip_df, prev_df, left_index=True, right_index=True)
    
    trip_df.columns = trip_df.columns.get_level_values(0)
    trip_df.columns = ['end', 'start_next', 'end_lat', 'end_long', 'end_prev', 'start', 'start_lat', 'start_long']
    trip_df['car'] = car_num
    return trip_df
trips = pd.DataFrame()

for car in tqdm.tqdm(np.array(pivot_df.columns.get_level_values(1))):
    trips = trips.append(get_car_trips(pivot_df, car))
    
def trip_distance(lat1, lat2, lon1, lon2):
    try:
        coords_1 = (lat1, lon1)
        coords_2 = (lat2, lon2)
        return geopy.distance.vincenty(coords_1, coords_2).km
    except ValueError:
        return -1
    
trips['trip_len'] = trips.apply(lambda x: trip_distance(x.start_lat, x.end_lat, x.start_long, x.end_long), axis=1)
trips.reset_index(inplace=True)


def transform_time(x):
    try:
        return datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S')
    except TypeError:
        return -1
    
trips['end'] = trips.end.apply(transform_time)
trips['start'] = trips.start.apply(transform_time)


trips = trips[trips.trip_len > 0.5]
trips['trip_duration'] = trips.apply(lambda x: (x.end - x.start).seconds/60, axis=1)
trips = trips[trips.trip_duration > 3]

trips['start_hour'] = trips.start.apply(lambda x: x.hour)
trips['day'] = trips.start.apply(lambda x: x.date())
plt.figure(figsize=(8, 6))
plt.style.use('fivethirtyeight')
sns.distplot(trips.trip_duration, bins=np.linspace(0, 120, 60), kde=False)
plt.xlabel('Duration [mins]')

plt.figure(figsize=(8, 6))
sns.distplot(trips.trip_len, bins=np.linspace(0, 12, 60), kde=False)
plt.xlim([1, 12])
plt.xlabel('Distance [km]')
kmeans = KMeans(n_clusters=6)
trips['start_cluster'] = kmeans.fit_predict(trips[['start_lat', 'start_long']])
trips['end_cluster'] = kmeans.predict(trips[['end_lat', 'end_long']])
trips.head()
def aggregate_trips(trips, kmeans):
    f = {'trip': 'count', 'start': 'first'}
    grouped_trips = trips.groupby(['start_cluster', 'end_cluster', 'start_hour']).agg(f).reset_index()

    clusters = pd.DataFrame(kmeans.cluster_centers_)
    clusters.columns = ['lat', 'long']

    grouped_trips = pd.merge(grouped_trips, clusters, left_on='start_cluster', right_index=True)
    grouped_trips = pd.merge(grouped_trips, clusters, left_on='end_cluster', right_index=True)
    return grouped_trips


def group_by_hour(grouped_trips):
    grouped_hour = grouped_trips.groupby(['start_cluster', 'end_cluster', 'start_hour']).agg({'start': 'min', 'lat_x': 'first', 'long_x': 'first',
                                    'lat_y': 'first', 'long_y': 'first', 'trip': 'sum'}).reset_index()
    grouped_hour['date'] = [datetime.strptime('2019-01-01', '%Y-%m-%d')]*len(grouped_hour)
    grouped_hour['date'] = grouped_hour.apply(lambda x: x['date'] + timedelta(hours=x.start_hour), axis=1)
    return grouped_hour


def create_map(grouped_hour, weight):
    m = folium.Map(location=[32.13,34.8],zoom_start=12, tiles="CartoDB dark_matter")

    lines = [
        {
            'coordinates': [
                [grouped_hour.long_x.iloc[index], grouped_hour.lat_x.iloc[index]],
                [grouped_hour.long_y.iloc[index], grouped_hour.lat_y.iloc[index]],
            ],
            'dates': [
            str(grouped_hour['date'].iloc[index]),
            str(grouped_hour['date'].iloc[index])
            ],
            'color': 'gold',
            'weight': int(grouped_hour.trip.iloc[index])
    # 

        }
        for index in range(len(grouped_hour))
    ]

    features = [
        {
            'type': 'Feature',
            'geometry': {
                'type': 'LineString',
                'coordinates': line['coordinates'],
            },
            'properties': {
                'times': line['dates'],
                'style': {
                    'color': line['color'],
                    'weight': weight*line['weight']**1.5 if 'weight' in line else 5
                }
            }
        }
        for line in lines
    ]


    plugins.TimestampedGeoJson({
        'type': 'FeatureCollection',
        'features': features,
    }, period='PT1H', duration='PT1H', add_last_point=True).add_to(m)
    return m

# trips = trips.sample(40000)
# data = [trips[trips['start_hour']==sorted(trips['start_hour'].unique())[i]][['start_lat','start_long']].values.tolist() 
#         for i in range(len(trips['start_hour'].unique()))]

# monthDict = { 0:'00:00', 1:'01:00', 2:'02:00', 3:'03:00', 4:'04:00', 5:'05:00', 6:'06:00', 
#             7:'07:00', 8:'08:00', 9:'09:00', 10:'10:00', 11:'11:00', 12:'12:00',
#             13:'13:00', 14:'14:00', 15:'15:00', 16:'16:00', 17:'17:00', 18:'18:00', 
#             19:'19:00', 20:'20:00', 21:'21:00', 22:'22:00', 23:'23:00', 24:'24:00'}

# index = [monthDict[i] for i in sorted(trips['start_hour'].unique())]

# m = folium.Map(location=[32.13,34.8],zoom_start=12, tiles="CartoDB dark_matter")
# hm = plugins.HeatMapWithTime(data=data,index=index)

# hm.add_to(m)

# m
weekend = trips[trips.day.isin([datetime.strptime('2018-12-14', '%Y-%m-%d').date(), datetime.strptime('2018-12-15', '%Y-%m-%d').date(), 
                                datetime.strptime('2018-12-22', '%Y-%m-%d').date(), datetime.strptime('2018-12-21', '%Y-%m-%d').date(),
                             datetime.strptime('2018-12-29', '%Y-%m-%d').date(),  datetime.strptime('2018-01-28', '%Y-%m-%d').date(),
                                datetime.strptime('2019-01-05', '%Y-%m-%d').date(),  datetime.strptime('2019-01-04', '%Y-%m-%d').date()])]

week = trips[~trips.day.isin([datetime.strptime('2018-12-14', '%Y-%m-%d').date(), datetime.strptime('2018-12-15', '%Y-%m-%d').date(), 
                                datetime.strptime('2018-12-22', '%Y-%m-%d').date(), datetime.strptime('2018-12-21', '%Y-%m-%d').date(),
                             datetime.strptime('2018-12-29', '%Y-%m-%d').date(),  datetime.strptime('2018-01-28', '%Y-%m-%d').date(),
                                datetime.strptime('2019-01-05', '%Y-%m-%d').date(),  datetime.strptime('2019-01-04', '%Y-%m-%d').date()])]
agg_trips = aggregate_trips(week, kmeans)
grouped_by_hour = group_by_hour(agg_trips)
m = create_map(grouped_by_hour, 0.001)
m
agg_trips = aggregate_trips(weekend, kmeans)
grouped_by_hour = group_by_hour(agg_trips)
m = create_map(grouped_by_hour, 0.02)
m
