import pandas as pd
import numpy as np
#rides dataframe has rides data
rides_df = pd.read_csv('../input/los-angeles-metro-bike-share-trip-data/metro-bike-share-trip-data.csv')
#stations df has station level data
stations_df = pd.read_csv('../input/metro-bike-share-stations-2017-02-09.csv/metro-bike-share-stations-2017-02-09.csv')
rides_df.head(5)
stations_df.head(5)
#Go_live_date format changes? :(
rides_df.columns
#first, remove spaces in columns names, so it is easier to work with pandas 
rides_df.columns = [c.replace(' ', '_').lower() for c in rides_df.columns]
stations_df.columns = [c.replace(' ', '_').lower() for c in stations_df.columns]
#merge starting station name
rides_df = rides_df.rename(columns={'starting_station_id': 'station_id'})
rides_df = rides_df.merge(stations_df[['station_id', 'station_name', 'region']], on='station_id', how='left')
rides_df = rides_df.rename(columns={'station_id': 'starting_station_id',
                                    'station_name': 'station_name_start',
                                    'region': 'region_start'})
#merge ending station name
rides_df = rides_df.rename(columns={'ending_station_id': 'station_id'})
rides_df = rides_df.merge(stations_df[['station_id', 'station_name', 'region']], on='station_id', how='left')
rides_df = rides_df.rename(columns={'station_id': 'ending_station_id',
                                    'station_name': 'station_name_end',
                                    'region': 'region_end'})
rides_df.columns
rides_df.trip_id.nunique()
#just to check that table is unique at trip level
#and it is
rides_df.shape[0]
#does not exactly match, so i guess we could say we have only 67 stations in our dataset
rides_df.starting_station_id.nunique(), stations_df.shape[0]
#Duration column, as informed in https://bikeshare.metro.net/about/data/ is in minutes
#The average trip length is 1555.3 minutes (25.9 hours)???
#The median trip length is 600 minutes (10 hours)???
#These numbers strikes me quite a bit, how is this possible?
rides_df.duration.mean(), rides_df.duration.median()
#lets look again at the data
rides_df[['duration', 'start_time', 'end_time']].head(5)
#duration (in minuts) column
rides_df['duration_mins'] = rides_df.duration / 60.
rides_df.duration_mins.mean(), rides_df.duration_mins.median()
#Let's look at the data distribution...
rides_df.duration_mins.hist()
#lets remove entries with more than hours of duration
rides_clean = rides_df[rides_df.duration_mins < 180]
rides_clean.shape[0], rides_df.shape[0] - rides_clean.shape[0]
rides_clean.duration_mins.hist(bins=25)
rides_clean.duration_mins.quantile(q=[0, 0.01, 0.05])
#NA entries?
rides_clean[['starting_station_longitude', 'starting_station_latitude',\
                                            'ending_station_longitude','ending_station_latitude']].isna().sum()
#remove NA entries
rides_clean = rides_clean.loc[(rides_clean.starting_station_longitude.notna()) & \
                              (rides_clean.ending_station_longitude.notna())]
#Entries with lon / lat set to 0?
(rides_clean[['starting_station_longitude', 'starting_station_latitude',\
                                            'ending_station_longitude','ending_station_latitude']]==0).sum()
#station 4108 is either a station located here: https://www.latlong.net/c/?lat=0.000000&long=0.000000
#or it is some kind of auxilair station
rides_clean.loc[rides_clean.starting_station_longitude==0, 'starting_station_id'].unique()
stations_df.loc[stations_df.station_id == 4108]
#Remove entries starting or ending in the warehouse
rides_clean = rides_clean.loc[~((rides_clean.starting_station_id == 4108) \
                                | (rides_clean.ending_station_id == 4108))]
#Haversine distance
#from https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
from math import sin, cos, sqrt, atan2, radians

def hav_distance(row):
    """
    Calculates the haversine distance, given a pandas row with the following row format:
    lon1, lat1, lon2, lat2
    
    Returns the distance in KM.
    """
    # approximate radius of earth in km
    R = 6373.0

    dlon = radians(row[2]) - radians(row[0])
    dlat = radians(row[3]) - radians(row[1])

    a = sin(dlat / 2)**2 + cos(radians(row[1])) * cos(radians(row[3])) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
rides_clean['trip_distance'] = rides_clean[['starting_station_longitude', 'starting_station_latitude',\
                                            'ending_station_longitude','ending_station_latitude']]\
                                          .apply(hav_distance, axis=1)
#distance distribution
rides_clean.trip_distance.hist()
rides_clean.loc[rides_clean.duration_mins < 3].trip_distance.head(10)
rides_clean.loc[rides_clean.trip_distance == 0].duration_mins.hist()
rides_clean.loc[rides_clean.trip_distance == 0].trip_route_category.value_counts()
#mislabeled entry?
#I don't know what is the round trip definition, but this looks quite round trip to me
rides_clean.loc[rides_clean.trip_distance == 0].loc[rides_clean.trip_route_category=='One Way'][['starting_station_id', 'ending_station_id']]
rides_clean = rides_clean.loc[~((rides_clean.duration_mins <= 2) & (rides_clean.trip_distance == 0))]
rides_clean.shape[0], rides_df.shape[0] - rides_clean.shape[0]
#we have filtered out 3.5K rides from since the beginning...
import seaborn as sns
sns.jointplot(x=rides_clean.trip_distance,
              y=rides_clean.duration_mins,kind='hex', 
              size = 7, 
              joint_kws=dict(gridsize=200),
              xlim=(0, 5),
              ylim=(0, 30))
#calculate avg speed
#Initialized to 0, so that when we are in a Round trip we do not get a divided by 0 
rides_clean['avg_speed'] = 0
#one way indeces
ow_idx = rides_clean.loc[rides_clean.trip_route_category != 'Round Trip'].index
rides_clean.loc[ow_idx, 'avg_speed'] = rides_clean.loc[ow_idx, 'trip_distance'] / (rides_clean.loc[ow_idx, 'duration_mins'] / 60.0)
rides_clean.loc[ow_idx].avg_speed.mean()
#format is ISO 8601 (info from the webpage)
rides_clean['start_time'].head(5)
#as datetime
rides_clean['start_time_dt'] = pd.to_datetime(rides_clean.start_time)
rides_clean['end_time_dt'] = pd.to_datetime(rides_clean.end_time)
rides_clean[['start_time', 'start_time_dt']].head(5)
#the transformation looks all right
#Extract year, month, day, and day of the week from the datetime column 
rides_clean['year'] = rides_clean.start_time_dt.dt.year
rides_clean['month'] = rides_clean.start_time_dt.dt.month
rides_clean['day'] = rides_clean.start_time_dt.dt.day
#The day of the week with Monday=0, Sunday=6
rides_clean['dayweek'] = rides_clean.start_time_dt.dt.dayofweek
rides_clean.start_time_dt.min()
rides_clean.groupby(['year', 'month'])['trip_id'].count().plot.bar(figsize = (8, 6))
rides_clean.groupby(['day'])['trip_id'].count().plot.bar(figsize = (8, 6))
rides_clean.groupby(['dayweek'])['trip_id'].count().plot.bar(figsize = (8, 6))
rides_clean.groupby(['station_name_start'])['trip_id'].count().sort_values().plot.barh(figsize = (9, 12))
rides_clean.groupby(['station_name_end'])['trip_id'].count().sort_values().plot.barh(figsize = (9, 12))
rides_clean['route'] = rides_clean.station_name_start + '-->' + rides_clean.station_name_end
#we have 3585 routes, lets plot top 50
rides_clean.route.nunique()
rides_clean.groupby(['route'])['trip_id'].count().sort_values()[-55:].plot.barh(figsize = (9, 14))
#we will use folium, nice package for geographical data
import folium

def get_base_map(rides_clean):
    return folium.Map(location=[rides_clean.starting_station_latitude.mean(),
                             rides_clean.starting_station_longitude.mean()],
                   zoom_start=14,
                   tiles='cartodbpositron')

#base map
base = get_base_map(rides_clean)
base
def add_station_markers(initial_map, rides_clean):
    #station location visualization
    statdf = rides_clean.drop_duplicates(subset=['starting_station_id'])
    out_map = initial_map
    for lat, lon, name in zip(statdf['starting_station_latitude'], statdf['starting_station_longitude'], statdf['station_name_start']):
        out_map.add_child(folium.Marker(location=[lat,lon], popup=(folium.Popup(name))))
    return out_map

markers = add_station_markers(base, rides_clean)
markers
#groupby route
route_cnt = pd.DataFrame(rides_clean.groupby(['station_name_start', 'station_name_end', 'route'])['trip_id'].count()).reset_index()
route_cnt_swap = route_cnt.rename(columns = {'station_name_start': 'station_name_end',
                                             'station_name_end': 'station_name_start'})
#merge data for the routes, taking into account both directions
route_cnt_all = route_cnt.merge(route_cnt_swap, on=['station_name_start', 'station_name_end'])
#sum the trips count if it has two directions (do not sum twice round trips)
route_cnt_all['route_count'] = route_cnt_all['trip_id_x']
update_idx = route_cnt_all.station_name_start != route_cnt_all.station_name_end
route_cnt_all.loc[update_idx, 'route_count'] = route_cnt_all[update_idx].trip_id_x + route_cnt_all[update_idx].trip_id_y 
route_cnt_all.head(5)
#final steps, and filter only those with a volume > 200
route_cnt_all = route_cnt_all.rename(columns = {'route_x': 'route'})
routedf = rides_clean.drop_duplicates(subset=['route'])
routedf = routedf.merge(route_cnt_all, on = 'route')
routedf_vol = routedf.loc[routedf.route_count > 200]
def draw_route_lines(initial_map, routedf_vol):
    out_map = initial_map
    #draw every route as a line
    for ix, row  in routedf_vol[['starting_station_latitude', 'starting_station_longitude', 'ending_station_latitude', 'ending_station_longitude', 'route_count']].iterrows():
        #draw a line for every route
        folium.PolyLine([(row['starting_station_latitude'], row['starting_station_longitude']), \
                         (row['ending_station_latitude'], row['ending_station_longitude'])],
                         weight = row['route_count'] / 150.,
                         opacity = float(row['route_count']) / routedf_vol['route_count'].max(),
                         popup= str(int(row['route_count'])))\
              .add_to(out_map)
    return out_map

draw_route_lines(markers, routedf_vol)
#I would like to have done something like this:
#https://python-graph-gallery.com/300-draw-a-connection-line/
#but it did not seem so simple with folium :(
from folium import plugins
#mostly taken from https://alysivji.github.io/getting-started-with-folium.html
# convert to (n, 2) nd-array format for heatmap
#limited to 40K obs, as with more than that it dies
stationArr = rides_clean[['starting_station_latitude', 'starting_station_longitude']][:40000].as_matrix()
# plot heatmap
# - get base map
# - then draw the actual HeatMap
get_base_map(rides_clean).add_child(plugins.HeatMap(stationArr, radius=40, max_val=300))
#same as before, but adding route lines
(draw_route_lines(get_base_map(rides_clean), routedf_vol)).add_child(plugins.HeatMap(stationArr, radius=40, max_val=300))
#Extrat the hour from the start time
rides_clean['hour'] = rides_clean.start_time_dt.dt.hour
#pick top 15 starting stations 
top_15_stations = rides_clean.groupby(['station_name_start'])['trip_id']\
                             .count().sort_values(ascending=False)[:15]\
                             .index.values.tolist()
#assignt a rank for each of the stations
rank = [(top_15_stations[i], i + 1) for i in range(len(top_15_stations))]
#filter only rides that belong to top stations
rides_filt = rides_clean.loc[rides_clean.station_name_start.isin(top_15_stations)]
#capture the hourly count by station
rides_hourly = rides_filt.groupby(['hour', 'station_name_start'])['station_name_start'].count()
rides_hourly = pd.DataFrame(rides_hourly)
rides_hourly = rides_hourly.rename(columns={'station_name_start': 'count'})
rides_hourly = rides_hourly.reset_index()

#get the global station trips rank
rides_hourly['rank'] = rides_hourly.station_name_start.apply(lambda x: [r[1] for r in rank if r[0] == x][0])
#prepare the actual plot
#inspiration from https://www.kaggle.com/aashita/guide-to-animated-bubble-charts-using-plotly
#awesome package by 
from bubbly.bubbly import bubbleplot 
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
figure = bubbleplot(dataset=rides_hourly, x_column='rank', y_column='count', 
    bubble_column='station_name_start', time_column='hour', size_column='count', 
    x_title="Rank", y_title="Hourly count", title='Stations hourly trips', scale_bubble=3, height=650)


figure['layout']['xaxis']['range'] = [0, 16]
figure['layout']['yaxis']['range'] = [-20, 1500]
iplot(figure, config={'scrollzoom': True})
#most of it from
#https://plot.ly/python/animations/

figure = bubbleplot(dataset=rides_hourly, x_column='rank', y_column='count', 
    bubble_column='station_name_start', time_column='hour', size_column='count', 
    x_title="Rank", y_title="Hourly count", title='Stations hourly trips', scale_bubble=3, height=650)

sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Hour:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

# make frames
for hour in rides_hourly.hour.unique():
    for station in rides_hourly.station_name_start.unique():
        dataset_by_hour_and_station = rides_hourly.loc[(rides_hourly.hour == hour) & (rides_hourly.station_name_start == station)]
    slider_step = {'args': [
        [hour],
        {'frame': {'duration': 300, 'redraw': False},
         'mode': 'immediate',
       'transition': {'duration': 300}}
     ],
     'label': str(hour),
     'method': 'animate'}
    sliders_dict['steps'].append(slider_step)

figure['layout']['sliders'] = [sliders_dict]   
    
figure['layout']['xaxis']['range'] = [0, 16]
figure['layout']['yaxis']['range'] = [-20, 1500]
iplot(figure, config={'scrollzoom': True})
#generate a 'weight' count columns
rides_hourly['count_w'] = rides_hourly['count'] / rides_hourly['count'].max()
#merge lat/lon to this df
rides_hourly = rides_hourly.merge(rides_clean.drop_duplicates(subset='station_name_start')[['station_name_start', 'starting_station_latitude', 'starting_station_longitude']], on='station_name_start')
#? plugins.HeatMapWithTime
## Need to pass a list of lists!
h_out = []
for h in rides_hourly.hour.unique():
    list_h = rides_hourly.loc[rides_hourly.hour == h][['starting_station_latitude', 'starting_station_longitude', 'count_w']].values.tolist()
    h_out.append(list_h)
get_base_map(rides_clean).add_child(plugins.HeatMapWithTime(h_out, auto_play=True, radius=60, gradient={.2: 'blue', .4: 'lime', .6: 'red'}))
#get_base_map(rides_clean).add_child(plugins.HeatMapWithTime(h_out, auto_play=True, radius=60))