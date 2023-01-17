import numpy as np
import pandas as pd 
import numpy
import matplotlib.pyplot as plt
import geopy.distance
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

OD_2017 = pd.read_csv('../input/OD_2017.csv', low_memory=False, index_col=0);
Stations_2017 = pd.read_csv('../input/Stations_2017.csv', low_memory=False);
Elevation = pd.read_csv('../input/POINT_DATA.csv', low_memory=False);
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

OD_2017.head()

Stations_2017.head()
Elevation.head()
Elevation.columns =['longitude','latitude','elevation']
Elevation.head()
#Add column
Stations_2017['elevation'] = 'e'
#Work with copy of data
Stations_2017_Elevations= Stations_2017.copy()
Stations_2017_Elevations.head()
for index, row in Stations_2017.iterrows(): 
    latitude = str(row['latitude'])
    longitude = str(row['longitude'])
    query = Elevation.query('longitude < '+longitude+' + 0.0001 & longitude > '+longitude+' - 0.0001 & latitude < '+latitude+' + 0.0001 & latitude > '+latitude+' - 0.0001')
   
    rows, cols = query.shape
    if(rows ==1):
        Stations_2017.loc[index, 'elevation'] = query['elevation'].iloc[0]
Stations_2017.head(10)
Stations_2017.query('elevation == "e"')
Stations_2017.to_csv(path_or_buf='Stations_2017_elevations.csv')
Stations_2017_Elevations= pd.read_csv('../input/Stations_2017_elevations_with_manually_added_missing_elevations.csv', low_memory=False,encoding = "cp1252");
#new Elevations with missing values added
Stations_2017_Elevations.head(100)
Stations_2017_Elevations.isnull().sum()

plt.figure(figsize=(15,5));
plt.subplot(1,2,1);
Stations_2017_Elevations['elevation'].plot.hist(bins=40);
plt.xlabel('Elevation');

df_complete = OD_2017.sort_values(by = ['start_station_code'])
Stations_2017_Elevations.sort_values(by = ['code'])

df_complete = pd.merge(df_complete, Stations_2017_Elevations,  how='left', left_on = 'start_station_code', right_on = 'code')
df_complete.head(100)
df_complete.rename(columns={'latitude': 'latitude_start', 'longitude': 'longitude_start','elevation': 'elevation_start','name': 'name_start', 'is_public': 'is_public_start'}, inplace=True)
df_complete.drop(columns=['code'],axis=1, inplace=True)
df_complete.head()
df_complete.loc[df_complete['end_station_code'] == 'Tabletop (RMA)']
# Remove data that doesn't match rest of data
df_complete = df_complete[df_complete.end_station_code !='Tabletop (RMA)']
#cast data to number
df_complete["end_station_code"] = pd.to_numeric(df_complete["end_station_code"]) 
df_complete_end_elevation = pd.merge(df_complete, Stations_2017_Elevations,  how='left', left_on = 'end_station_code', right_on = 'code')
df_complete_end_elevation.head()
df_complete_end_elevation.rename(columns={'latitude': 'latitude_end', 'longitude': 'longitude_end','name': 'name_end','elevation': 'elevation_end','name': 'name_end', 'is_public': 'is_public_end'}, inplace=True)
df_complete_end_elevation.drop(columns=['code'],axis=1, inplace=True)
df_complete_end_elevation.head()
df_complete_end_elevation['elevation_difference'] = df_complete_end_elevation.apply(lambda row: row.elevation_end - row.elevation_start, axis=1)
df_complete_end_elevation.head(5)
#df_complete_end_elevation['distance_between_stations'] = df_complete_end_elevation.apply(lambda row: geopy.distance.distance((row.latitude_start,row.longitude_start), (row.latitude_end ,row.longitude_end)), axis=1)
df_complete_end_elevation.shape
row = df_complete_end_elevation.iloc[1]
distance = geopy.distance.distance((row.latitude_start,row.longitude_start), (row.latitude_end ,row.longitude_end))
print(distance)
df_distances = pd.DataFrame(columns=['distance'])
df =df_complete_end_elevation.iloc[0:5]
df

#Takes 40 minutes to calculate on my laptop, no need to run as dataset with distance values are loaded later on

#for index, row in df_complete_end_elevation.iterrows():
#    df_complete_end_elevation.set_value(index,'distance',geopy.distance.distance((row.latitude_start,row.longitude_start), (row.latitude_end ,row.longitude_end)))
#    if(index%10000==0):
#        print(index)
#        
#df_complete_end_elevation.head(10)

#df_complete_end_elevation.to_csv('complete_data_elevations_and_distance.csv')
df_bixi = pd.read_csv('../input/complete_data_elevations_and_distance.csv', low_memory=False);

df_bixi.info()
# change object distance with end km to float
df_bixi['distance'] = df_bixi['distance'].astype(str).str[:-3].astype(float)
df_bixi.info()

df_bixi.head()
plt.figure(figsize=(20,10));
plt.subplot(1,2,1);
df_bixi['distance'].plot.hist(bins=40)
plt.xlabel('Distance Km');


plt.subplot(1,2,2);
df_bixi['elevation_difference'].plot.hist(bins=100)
plt.xlabel('Elevation Difference (m)');

fig = plt.figure(figsize=(10,10));
ax = fig.gca()
ax.set_xticks(numpy.arange(0, 30, 2))
ax.set_yticks(numpy.arange(-50, 50, 2))
plt.grid()
plt.boxplot(df_bixi['elevation_difference'], 0, '')


df_bixi.describe()
df_bixi.median()
df_bixi[df_bixi['elevation_difference'] <0 ].count()
#show data about elevation on specific days and times
df_bixi['start_date'] = pd.to_datetime(OD_2017['start_date'])
df_bixi['end_date'] = pd.to_datetime(OD_2017['end_date'])
df_bixi[['start_date', 'end_date']].dtypes
plt.figure(figsize=(15,5));
df_bixi['duration_sec'].plot.hist(bins=100);
plt.xlabel('Duration');


plt.subplot(1,2,1);
df_bixi.groupby('is_member').mean()['elevation_difference'].plot(kind='bar', color='#1f77b4');
plt.title('Mean Elevation Difference');
df_bixi['weekday'] = df_bixi.start_date.dt.dayofweek
df_bixi['hour'] = df_bixi.start_date.dt.hour
df_bixi['month'] = df_bixi.start_date.dt.month
df_bixi['daynum'] = df_bixi.start_date.dt.dayofyear
plt.figure(figsize=(15,5));
plt.subplot(1,3,1);
df_bixi.groupby('weekday').count()['duration_sec'].plot(kind='bar', color='#1f77b4');

plt.subplot(1,3,2);
df_bixi.groupby('hour').count()['duration_sec'].plot(kind='bar', color='#1f77b4');

plt.subplot(1,3,3);
df_bixi.groupby('month').count()['duration_sec'].plot(kind='bar', color='#1f77b4');
dfp = df_bixi.pivot_table(columns='hour',index='weekday', aggfunc=np.mean)['elevation_difference'];
plt.figure(figsize=(18,5));
plt.title('Pivot table: Mean Elevation Difference');
plt.imshow(dfp,interpolation='none');
hours = range(24);
hourlabels = map(lambda x: str(x)+'h',hours);
days = range(7);
daylabels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
plt.xticks(hours,hourlabels,rotation=90);
plt.yticks(days,daylabels);
plt.colorbar();
dfp = df_bixi.pivot_table(columns='hour',index='weekday', aggfunc=np.mean)['distance'];
plt.figure(figsize=(18,5));
plt.title('Pivot table: Trip Distance (km)');
plt.imshow(dfp,interpolation='none');
hours = range(24);
hourlabels = map(lambda x: str(x)+'h',hours);
days = range(7);
daylabels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
plt.xticks(hours,hourlabels,rotation=90);
plt.yticks(days,daylabels);
plt.colorbar();
Stations_2017_Elevations['Coordinates']  = list(zip(Stations_2017_Elevations.longitude, Stations_2017_Elevations.latitude))
Stations_2017_Elevations['Coordinates'] = Stations_2017_Elevations['Coordinates'].apply(Point)
gdf = gpd.GeoDataFrame(Stations_2017_Elevations, geometry='Coordinates')
print(gdf.head())
gdf.plot()

vmin, vmax = -10, 100

ax = gdf.plot(column='elevation', colormap='hot', vmin=vmin, vmax=vmax)

# add colorbar
fig = ax.get_figure()
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=vmin, vmax=vmax))
# fake up the array of the scalar mappable. Urgh...
sm._A = []
fig.colorbar(sm, cax=cax)






