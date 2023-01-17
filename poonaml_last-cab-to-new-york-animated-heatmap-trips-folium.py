#Let's check the preps, food check Beer check venue check!!

import matplotlib.pyplot as plt

%matplotlib inline



from mpl_toolkits.basemap import Basemap

from matplotlib import cm



import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)

import pandas as pd

pd.options.mode.chained_assignment = None 

import numpy as np



from haversine import haversine

from scipy.spatial.distance import euclidean , cityblock

from geopy.distance import great_circle

from math import *



from bokeh.io import output_notebook,show

from bokeh.models import HoverTool

from bokeh.plotting import figure

from bokeh.palettes import Spectral4



import folium 

from folium import plugins

from folium.plugins import HeatMap



output_notebook()
!unzip ../input/nyc-taxi-trip-duration/train.zip

!unzip ../input/nyc-taxi-trip-duration/test.zip
!ls 
#bring in the 6 packs

train=pd.read_csv("train.csv")#../input/nyc-taxi-trip-duration/

train.head(3)

#Verify assumptions and loopholes in our analysis

test=pd.read_csv("test.csv")#../input/nyc-taxi-trip-duration/

test.head(3)
west, south, east, north = -74.03, 40.63, -73.77, 40.85



train = train[(train.pickup_latitude> south) & (train.pickup_latitude < north)]

train = train[(train.dropoff_latitude> south) & (train.dropoff_latitude < north)]

train = train[(train.pickup_longitude> west) & (train.pickup_longitude < east)]

train = train[(train.dropoff_longitude> west) & (train.dropoff_longitude < east)]
#Extract the month column from pickup datetime variable and take subset of data

train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

train['dropoff_month'] = train['dropoff_datetime'].dt.month



heat_df =train.sample(n=2500)



#Extract required columns

heat_df = heat_df[['dropoff_latitude', 'dropoff_longitude','dropoff_month']]





# Ensure you're handing it floats

heat_df['dropoff_latitude'] = heat_df['dropoff_latitude'].astype(float)

heat_df['dropoff_longitude'] = heat_df['dropoff_longitude'].astype(float)





#remove NANs

heat_df = heat_df.dropna(axis=0)





# Create weight column, using date

heat_df['Weight'] = heat_df['dropoff_month']

heat_df['Weight'] = heat_df['Weight'].astype(float)

heat_df = heat_df.dropna(axis=0, subset=['dropoff_latitude','dropoff_longitude', 'Weight'])
newyork_on_heatmap = folium.Map(location=[40.767937,-73.982155 ],tiles= "Stamen Terrain",

                    zoom_start = 13) 



# List comprehension to make out list of lists

heat_data = [[[row['dropoff_latitude'],row['dropoff_longitude']] 

                for index, row in heat_df[heat_df['Weight'] == i].iterrows()] 

                 for i in range(0,6)]



# Plot it on the map

hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)

hm.add_to(newyork_on_heatmap)



# Display the map

newyork_on_heatmap
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(15,10))



train.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude',

                color='yellow', 

                s=.02, alpha=.6, subplots=True, ax=ax1)

ax1.set_title("Pickups")

ax1.set_facecolor('black')



train.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',

                color='yellow', 

                s=.02, alpha=.6, subplots=True, ax=ax2)

ax2.set_title("Dropoffs")

ax2.set_facecolor('black') 
#co-ordinates

LaGuardia = {

    "minLat": 40.76,

    "maxLat": 40.78,

    "minLong": -73.895,

    "maxLong": -73.855

}

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)



LaGuardiaData = train[(train['pickup_longitude'].apply(lambda x: (x >=LaGuardia["minLong"]) & (x <= LaGuardia["maxLong"])))]

LaGuardiaData = train[(train['pickup_latitude'].apply(lambda x: (x >=LaGuardia["minLat"]) & (x <= LaGuardia["maxLat"])))]

LaGuardiaData = train[(train['dropoff_longitude'].apply(lambda x: (x >=LaGuardia["minLong"]) & (x <= LaGuardia["maxLong"])))]

LaGuardiaData = train[(train['dropoff_latitude'].apply(lambda x: (x >=LaGuardia["minLat"]) & (x <= LaGuardia["maxLat"])))]



m = folium.Map(

    location=[40.7769, -73.8740],

    zoom_start=12

)

folium.Marker(location=[40.7769, -73.8740],icon=folium.Icon(color='black') ,popup='LA Guardia International Airport').add_to(m)



shortTripsDF=LaGuardiaData[LaGuardiaData.trip_duration==900]



lines = [

    {

        'coordinates': [

            [shortTripsDF.pickup_longitude.iloc[index], shortTripsDF.pickup_latitude.iloc[index]],

            [shortTripsDF.dropoff_longitude.iloc[index], shortTripsDF.dropoff_latitude.iloc[index]],

        ],

        'dates': [

        str(shortTripsDF.pickup_datetime.iloc[index]),

        str(shortTripsDF.dropoff_datetime.iloc[index])

        ],

        'color': 'gold'

    }

    for index in range(100)

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

                'weight': line['weight'] if 'weight' in line else 10

            }

        }

    }

    for line in lines

]





plugins.TimestampedGeoJson({

    'type': 'FeatureCollection',

    'features': features,

}, period='PT24H', add_last_point=True).add_to(m)

m
pd.DataFrame({'Train': [train.shape[0]], 'Test': [test.shape[0]]}).plot.barh(

    figsize=(15, 2), legend='reverse',  color=["black","gold"])

plt.title("Number of examples in each data set")

plt.ylabel("Data sets")

plt.yticks([])

plt.xlabel("Number of examples");
print("Training headcount is %i." % train.shape[0])

print("Testing headcount is %i." % test.shape[0])
pd.DataFrame({'Train': [train.shape[1]], 'Test': [test.shape[1]]}).plot.barh(

    figsize=(15, 2), legend='reverse',  color=["black","gold"])

plt.title("Number of columns in each data set")

plt.ylabel("Data sets")

plt.yticks([])

plt.xlabel("Number of columns");
pd.set_option('display.float_format', lambda x: '%.2f' % x)

train.describe()
test.describe()
plt.scatter(train.trip_duration,train.index,color="gold")

plt.xlabel("Trip Duration")

plt.title("Trip Duration for each Taxi ride");
train['log_trip_duration'] = np.log1p(train['trip_duration'].values)
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,8))

fig.suptitle('Train trip duration and log of trip duration')

ax1.legend(loc=0)

ax1.set_ylabel('count')

ax1.set_xlabel('trip duration')

ax2.set_xlabel('log(trip duration)')

ax2.legend(loc=0)

ax1.hist(train.trip_duration,color='black',bins=7)

ax2.hist(train.log_trip_duration,bins=50,color='gold');
print("Skewness: %f" % train['log_trip_duration'].skew())

print("Kurtosis: %f" % train['log_trip_duration'].kurt())
train["vendor_id"].value_counts().plot(kind='bar',color=["black","gold"])

plt.xticks(rotation='horizontal')

plt.title("Vendors")

plt.ylabel("Count for each Vender")

plt.xlabel("Vendor Ids");
train["passenger_count"].value_counts().plot(kind='bar',color=["black","gold"])

plt.title("Passengers in a group of")

plt.xticks(rotation='horizontal')

plt.ylabel("Count for each passenger")

plt.xlabel("Number of Passengers");
train["store_and_fwd_flag"].value_counts().plot(kind='bar',color=["black","gold"])

plt.title("Store and Forward cases")

plt.xticks(rotation='horizontal')

plt.ylabel("Count for flags")

plt.xlabel("Flag");
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)





for df in (train,test):

    # Dates

    df['pickup_date'] = df['pickup_datetime'].dt.date



    # day of month 1 to 30/31

    df['pickup_day'] = df['pickup_datetime'].dt.day



    #month of year 1 to 12

    df['pickup_month'] = df['pickup_datetime'].dt.month



    #weekday 0 to 6

    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday





    #week of year

    df['pickup_weekofyear'] = df['pickup_datetime'].dt.weekofyear



    #hour of day 0 to 23

    df['pickup_hour'] = df['pickup_datetime'].dt.hour



    #minute of hour

    df['pickup_minute'] = df['pickup_datetime'].dt.minute



    # day of year

    df['pickup_dayofyear'] = df['pickup_datetime'].dt.dayofyear



train['pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()

train['pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']





test['pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()

test['pickup_week_hour'] = test['pickup_weekday'] * 24 + test['pickup_hour']

plt.figure(figsize=(15, 6)) 

train.pickup_month.value_counts().plot(kind='bar',color=["black","gold"],align='center',width=0.3)

plt.xticks(rotation='horizontal')

plt.xlabel("months")

plt.ylabel("Number of trips")

plt.title("Total trips in each month");
sns.swarmplot(train.pickup_month[:1000],train.log_trip_duration[:1000],hue=train.vendor_id[:1000],palette={1:'gold',2:'black'});
tripsByDate=train['pickup_date'].value_counts()



# Basic plot setup

plot = figure( x_axis_type="datetime", tools="",

              toolbar_location=None, x_axis_label='Dates',

            y_axis_label='Taxi trip counts', title='Hover over points to see taxi trips')



x,y= tripsByDate.index, tripsByDate.values

plot.line(x,y, line_dash="4 4", line_width=1, color='gold')



cr = plot.circle(x, y, size=20,

                fill_color="gold", hover_fill_color="black",

                fill_alpha=0.05, hover_alpha=0.5,

                line_color=None, hover_line_color="black")

plot.left[0].formatter.use_scientific = False



plot.add_tools(HoverTool(tooltips=None, renderers=[cr], mode='hline'))



show(plot)
print("Highest number of pickups  i.e", tripsByDate[0] , "happened on", str(tripsByDate.index[0]))

print("And lowest number of pickups  i.e", tripsByDate[tripsByDate.size-1] , "happened on",

      str(tripsByDate.index[tripsByDate.size-1]), "due to heavy snowfall in New York.")
snowFallDF=train[(train['pickup_dayofyear'] == 24) |  (train['pickup_dayofyear'].any() == 23)]

snowFallDF.shape
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(15,10))



snowFallDF.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude',

                color='yellow', 

                s=.02, alpha=.6, subplots=True, ax=ax1)

ax1.set_title("Pickups")

ax1.set_facecolor('black')



snowFallDF.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',

                color='yellow', 

                s=.02, alpha=.6, subplots=True, ax=ax2)

ax2.set_title("Dropoffs")

ax2.set_facecolor('black') 
train['dropoff_date'] = train['dropoff_datetime'].dt.date

tripsByDropoffDate=train['dropoff_date'].value_counts()



# Basic plot setup

plot = figure( x_axis_type="datetime", tools="",

              toolbar_location=None, x_axis_label='Dates',

            y_axis_label='Taxi trip counts', title='Hover over points to see taxi trips')



x,y= tripsByDropoffDate.index, tripsByDropoffDate.values

plot.line(x,y, line_dash="4 4", line_width=1, color='gold')



cr = plot.circle(x, y, size=20,

                fill_color="gold", hover_fill_color="black",

                fill_alpha=0.05, hover_alpha=0.5,

                line_color=None, hover_line_color="black")

plot.left[0].formatter.use_scientific = False



plot.add_tools(HoverTool(tooltips=None, renderers=[cr], mode='hline'))



show(plot)

train.drop('dropoff_date',axis=1,inplace=True)
print("Highest number of dropoffs  i.e", tripsByDropoffDate[0] , "happened on", str(tripsByDropoffDate.index[1]))

print("And lowest number of dropoffs  i.e", tripsByDropoffDate[tripsByDropoffDate.size-1] , "happened on",

      str(tripsByDropoffDate.index[tripsByDropoffDate.size-1]))
plt.figure(figsize=(15, 6)) 

train.pickup_day.value_counts().plot(kind='bar',color=["black","gold"],align='center',width=0.3)

plt.xlabel("Days")

plt.xticks(rotation='horizontal')

plt.ylabel("Number of trips")

plt.title("Total trips on each day");
sns.regplot(train.pickup_day[:1000],train.log_trip_duration[:1000],color='gold', line_kws={'color':'black'});
train['pickup_weekday_name'] = train['pickup_datetime'].dt.weekday_name

plt.figure(figsize=(15, 6)) 

train.pickup_weekday_name.value_counts().plot(kind='bar',color=["black","gold"],align='center',width=0.3)

plt.xlabel("WeekDays")

plt.xticks(rotation='horizontal')

plt.ylabel("Number of trips")

plt.title("Total trips on each weekday");

train.drop('pickup_weekday_name',axis=1,inplace=True)
plt.figure(figsize=(15, 6)) 

train.pickup_hour.value_counts().plot(kind='bar',color=["black","gold"],align='center',width=0.3)

plt.xlabel("Hour")

plt.xticks(rotation='horizontal')

plt.ylabel("Number of trips")

plt.title("Total pickups at each hour");
sns.factorplot(x="pickup_hour", y="log_trip_duration", data=train,color='gold',size=7);
train['dropoff_hour'] = train['dropoff_datetime'].dt.hour

plt.figure(figsize=(15, 6)) 

train.dropoff_hour.value_counts().plot(kind='bar',color=["black","gold"],align='center',width=0.3)

plt.xticks(rotation='horizontal')

plt.xlabel("Hour")

plt.ylabel("Number of trips")

plt.title("Total dropoffs at each hour");

train.drop('dropoff_hour',axis=1,inplace=True)
fig=plt.figure(figsize=(15,10))

sns.violinplot(x="pickup_minute", y="log_trip_duration", data=train[:1000],color='black');
train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')

test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')
fig, ax = plt.subplots(ncols=2, nrows=2,figsize=(12, 12), sharex=False, sharey = False)

ax[0,0].hist(train.pickup_latitude.values,bins=40,color="gold")

ax[0,1].hist(train.pickup_longitude.values,bins=35,color="black")

ax[1,0].hist(train.dropoff_latitude.values,bins=40,color="gold")

ax[1,1].hist(train.dropoff_longitude.values,bins=35,color="black")

ax[0,0].set_xlabel('Pickup Latitude')

ax[0,1].set_xlabel('Pickup Longitude')

ax[1,0].set_xlabel('Dropoff Latitude')

ax[1,1].set_xlabel('Dropoff Longitude');
train=train[:100000]

test=test[:100000]



train['lat_diff'] = train['pickup_latitude'] - train['dropoff_latitude']

test['lat_diff'] = test['pickup_latitude'] - test['dropoff_latitude']



train['lon_diff'] = train['pickup_longitude'] - train['dropoff_longitude']

test['lon_diff'] = test['pickup_longitude'] - test['dropoff_longitude']
train['haversine_distance'] = train.apply(lambda row: haversine( (row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude']) ), axis=1)

test['haversine_distance'] = test.apply(lambda row: haversine( (row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude']) ), axis=1)

train['log_haversine_distance'] = np.log1p(train['haversine_distance']) 

test['log_haversine_distance'] = np.log1p(test['haversine_distance']) 
plt.scatter(train.log_haversine_distance,train.log_trip_duration,color="gold",alpha=0.04)

plt.ylabel("log(Trip Duration)")

plt.xlabel("log(Haversine Distance)")

plt.title("log(Haversine Distance) Vs log(Trip Duration)");
def manhattan_distance(x,y):

  return sum(abs(a-b) for a,b in zip(x,y))



def euclidean_distance(x,y):

  return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))



train['euclidean_distance'] = train.apply(lambda row: euclidean_distance( (row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude']) ), axis=1)

test['euclidean_distance'] = test.apply(lambda row: euclidean_distance( (row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude']) ), axis=1)

train['log_euclidean_distance'] = np.log1p(train['euclidean_distance']) 

test['log_euclidean_distance'] = np.log1p(test['euclidean_distance']) 



train['manhattan_distance'] = train.apply(lambda row: manhattan_distance( (row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude']) ), axis=1)

test['manhattan_distance'] = test.apply(lambda row: manhattan_distance( (row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude']) ), axis=1)

train['log_manhattan_distance'] = np.log1p(train['manhattan_distance']) 

test['log_manhattan_distance'] = np.log1p(test['manhattan_distance']) 
plt.scatter(train.log_manhattan_distance,train.log_trip_duration,color="black",alpha=0.04)

plt.ylabel("log(Trip Duration)")

plt.xlabel("log(Manhattan  Distance)")

plt.title("log(Manhattan Distance) Vs log(Trip Duration)");
train.loc[:, 'avg_speed_h'] = 1000 * train['haversine_distance'] / train['trip_duration']

train.loc[:, 'avg_speed_m'] = 1000 * train['manhattan_distance'] / train['trip_duration']

train.loc[:, 'avg_speed_eu'] = 1000 * train['euclidean_distance'] / train['trip_duration']



test.loc[:, 'avg_speed_h'] = 1000 * test['haversine_distance'] / train['trip_duration']

test.loc[:, 'avg_speed_m'] = 1000 * test['manhattan_distance'] / train['trip_duration']

test.loc[:, 'avg_speed_eu'] = 1000 * test['euclidean_distance'] / train['trip_duration']
fig, ax = plt.subplots(ncols=2, sharey=True)

ax[0].plot(train.groupby('pickup_hour').mean()['avg_speed_h'], '^', lw=2, alpha=0.7,color='black')

ax[1].plot(train.groupby('pickup_weekday').mean()['avg_speed_h'], 's', lw=2, alpha=0.7,color='gold')

ax[0].set_xlabel('hour')

ax[1].set_xlabel('weekday')

ax[0].set_ylabel('average speed')

fig.suptitle('Rush hour average traffic speed')

plt.show()
fastest_routes1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])

fastest_routes2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

test_fastest_routes = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv',

                               usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

train_fastest_routes = pd.concat((fastest_routes1, fastest_routes2))

train = train.merge(train_fastest_routes, how='left', on='id')

test = test.merge(test_fastest_routes, how='left', on='id')
target=train.log_trip_duration.values

train = train.drop(['id', 'pickup_datetime','dropoff_month', 'haversine_distance', 'manhattan_distance','euclidean_distance','avg_speed_h','avg_speed_m','avg_speed_eu','dropoff_datetime', 'trip_duration','log_trip_duration','pickup_date'], axis=1)

train.fillna(0,inplace=True)

train.dtypes

Id=test.id.values

test = test.drop(['id','pickup_datetime','pickup_date','haversine_distance', 'manhattan_distance','euclidean_distance','avg_speed_h','avg_speed_m','avg_speed_eu'], axis=1)

test.fillna(0,inplace=True)

predictors=test.columns

test.dtypes
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=1000, min_samples_leaf=50, min_samples_split=75)
rf_model.fit(train.values, target)

predictions=rf_model.predict(test.values)

predictions[:5]
test['trip_duration'] = np.exp(predictions) - 1

test['id']=Id

test[['id', 'trip_duration']].to_csv('poonam.csv.gz', index=False, compression='gzip')

test['trip_duration'][:5]
importances=rf_model.feature_importances_

std = np.std([rf_model.feature_importances_ for tree in rf_model.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

sorted_important_features=[]

for i in indices:

    sorted_important_features.append(predictors[i])

plt.figure(figsize=(10,10))

plt.title("Feature Importances By Random Forest Model")

plt.barh(range(len(indices)), importances[indices],

       color=["black","gold"], yerr=std[indices], align="center")

plt.yticks(range(len(indices)), sorted_important_features, rotation='horizontal');