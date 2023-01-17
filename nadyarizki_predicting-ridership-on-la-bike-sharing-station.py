import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 120)
import time
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder
import folium
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score 
import joblib
from sklearn.model_selection import GridSearchCV
dfq1 = pd.read_csv("../input/los-angeles-bike-trips-2019/metro-bike-share-trips-2019-q1.csv").sample(n=20000, random_state=1)
dfq2 = pd.read_csv("../input/los-angeles-bike-trips-2019/metro-bike-share-trips-2019-q2.csv").sample(n=20000, random_state=1)
dfq3 = pd.read_csv("../input/los-angeles-bike-trips-2019/metro-bike-share-trips-2019-q3.csv").sample(n=20000, random_state=1)
dfq4 = pd.read_csv("../input/los-angeles-bike-trips-2019/metro-bike-share-trips-2019-q4.csv").sample(n=20000, random_state=1)
dfq1.head(2)
dfq2.head(2)
dfq3.head(2)
dfq4.head(2)
dfq1['start_time'] = dfq1['start_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) 
dfq2['start_time'] = dfq2['start_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) 
dfq3['start_time'] = dfq3['start_time'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M')) 
dfq4['start_time'] = dfq4['start_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) 

dfq1['end_time'] = dfq1['end_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) 
dfq2['end_time'] = dfq2['end_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) 
dfq3['end_time'] = dfq3['end_time'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M')) 
dfq4['end_time'] = dfq4['end_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) 

dfq3['start_time'] = dfq3['start_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
dfq3['end_time'] = dfq3['end_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
dfq3['start_time'] = dfq3['start_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) 
dfq3['end_time'] = dfq3['end_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

df = pd.concat([dfq1, dfq2, dfq3, dfq4], axis = 0)
df.isna().sum()
startstat = []
for i in range(len(df[df['start_lat'].isnull()])):
       startstat.append(df[df['start_lat'].isnull()].iloc[i]['start_station'])
        
endstat = []
for i in range(len(df[df['end_lat'].isnull()])):
       endstat.append(df[df['end_lat'].isnull()].iloc[i]['end_station'])
        
print (set(endstat))
print (set(startstat))
dfst = pd.read_csv("../input/los-angeles-bike-trips-2019/metro-bike-share-stations-2020-01-01.csv")

print(dfst[dfst['Station_ID']==list(set(startstat))[0]]['Station_Name'].iloc[0])
print(dfst[dfst['Station_ID']==list(set(startstat))[1]]['Station_Name'].iloc[0])
print(dfst[dfst['Station_ID']==list(set(startstat))[2]]['Station_Name'].iloc[0])
df.dropna(axis =0, inplace=True)
print('amount of trips involving station 4327: start: ',len(df[df['start_station'] == 4327]), ' end: ', len(df[df['end_station'] == 4327]))
df = df[df['start_station']!=4327]
df = df[df['end_station']!=4327]

print('amount of trips involving station 4363: start: ',len(df[df['start_station'] == 4363]), ' end: ', len(df[df['end_station'] == 4363]))
df = df[df['start_station']!=4363]
df = df[df['end_station']!=4363]

print('amount of trips involving station 4108: start: ',len(df[df['start_station'] == 4108]), ' end: ', len(df[df['end_station'] == 4108]))
df = df[df['start_station']!=4108]
df = df[df['end_station']!=4108]

print('amount of trips involving station 4467: start: ',len(df[df['start_station'] == 4467]), ' end: ', len(df[df['end_station'] == 4467]))
df = df[df['start_station']!=4467]
df = df[df['end_station']!=4467]

print('amount of trips involving station 4468: start: ',len(df[df['start_station'] == 4468]), ' end: ', len(df[df['end_station'] == 4468]))
df = df[df['start_station']!=4468]
df = df[df['end_station']!=4468]
df['reg_start'] = df['start_station'].apply(lambda x: dfst[dfst['Station_ID']==x]['Region '].iloc[0])
df['reg_end'] = df['end_station'].apply(lambda x: dfst[dfst['Station_ID']==x]['Region '].iloc[0])
df['time'] = df['start_time'].apply(lambda x: x.hour)
# token = '**********'
# station_id = 'GHCND:USW00093134' # station Downtown Los Angeles
# baseurl = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data?'
# datasetid = 'GHCND'
# datatypeid0 = 'TMIN' ## minimum temperature
# datatypeid1 = 'TMAX' ## maximum temperature
# datatypeid2 = 'PRCP' ## percipitation
# url1 = baseurl+'datasetid='+datasetid+'&datatypeid='+datatypeid1+'&limit=1000&stationid='+station_id+'&startdate=2019-01-01&enddate=2019-12-31'
# url2 = baseurl+'datasetid='+datasetid+'&datatypeid='+datatypeid2+'&limit=1000&stationid='+station_id+'&startdate=2019-01-01&enddate=2019-12-31'
# url0 = baseurl+'datasetid='+datasetid+'&datatypeid='+datatypeid0+'&limit=1000&stationid='+station_id+'&startdate=2019-01-01&enddate=2019-12-31'
# headers={'token': token}

# data0 = requests.get(url0, headers={'token': token})
# data0 = data0.json()
# date0 = []
# tempmin = []
# for i in data0['results']:
#     tempmin.append(i['value']/10)
#     date0.append(i['date'])

# data1 = requests.get(url1, headers={'token': token})
# data1 = data1.json()
# data1
# date1 = []
# tempmax = []
# for i in data1['results']:
#     tempmax.append(i['value']/10)
#     date1.append(i['date'])
    
# data2 = requests.get(url2, headers={'token': token})
# data2 = data2.json()
# data2
# pcp = []
# date2 = []
# for i in data2['results']:
#     pcp.append(round(i['value']/10/25.4, 2))
#     date2.append(i['date'])
    
# dfw = pd.DataFrame()
# dfw['date'] = date1
# dfw['temp_max'] = tempmax
# dfw['temp_min'] = tempmin
# dfw['pcp'] = pcp
# dfw['date'] = dfw['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'))
# dfw['date'] = dfw['date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
# dfw['date'] = dfw['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

# dfw.to_csv('weather.csv')
dfw = pd.read_csv("../input/los-angeles-bike-trips-2019/weather.csv")
dfw['date'] = dfw['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
df['date'] = df['start_time'].apply(lambda x: x.date())
dfw['date'] = dfw['date'].apply(lambda x: x.date())
df['tempmax'] = df['date'].apply(lambda x: dfw[dfw['date'] == x]['temp_max'].iloc[0])
df['tempmin'] = df['date'].apply(lambda x: dfw[dfw['date'] == x]['temp_min'].iloc[0])
df['pcp'] = df['date'].apply(lambda x: dfw[dfw['date'] == x]['pcp'].iloc[0])
uniquestat = list (df['start_station'].unique()) + list(df['end_station'].unique())
uniquestat = list(set(uniquestat))
len(uniquestat)
statname = []
statlat = []
statlong = []
statid=[]
statreg = []

for i in list (df['start_station'].unique()): 
    statid.append(i)
    statname.append(dfst[dfst['Station_ID']==i]['Station_Name'].iloc[0])
    statlat.append(df[df['start_station']==i]['start_lat'].iloc[0])
    statlong.append(df[df['start_station']==i]['start_lon'].iloc[0])
    statreg.append(df[df['start_station']==i]['reg_start'].iloc[0])
    
for i in list (df['end_station'].unique()): 
    statid.append(i)
    statname.append(dfst[dfst['Station_ID']==i]['Station_Name'].iloc[0])
    statlat.append(df[df['end_station']==i]['end_lat'].iloc[0])
    statlong.append(df[df['end_station']==i]['end_lon'].iloc[0])
    statreg.append(df[df['end_station']==i]['reg_end'].iloc[0])

dfstat = pd.DataFrame()
dfstat['statid'] = statid
dfstat['name'] =statname
dfstat['statlat'] = statlat
dfstat['statlong'] = statlong
dfstat['statreg'] = statreg
dfstat = dfstat.drop_duplicates()
dfscore = pd.read_excel("../input/los-angeles-bike-trips-2019/StationScore.xlsx")
dfscore.head(3)
dfstat['walk_score'] =  dfstat['statid'].apply(lambda x: dfscore[dfscore['statid']==x]['Walk'].iloc[0])  
dfstat['transit_score'] =  dfstat['statid'].apply(lambda x: dfscore[dfscore['statid']==x]['Transit'].iloc[0])   
dfstat['bike_score'] =  dfstat['statid'].apply(lambda x: dfscore[dfscore['statid']==x]['Bike'].iloc[0])   
dfstat.isna().sum()
dfstat[dfstat['transit_score'].isnull()]
for i in dfstat[dfstat['transit_score'].isnull()].index:
    if dfstat.loc[i]['statreg']=='Westside':
        dfstat['transit_score'].loc[i] = dfstat[dfstat['statreg']=='Westside']['transit_score'].mean()
    elif dfstat.loc[i]['statreg']=='North Hollywood':
        dfstat['transit_score'].loc[i] = dfstat[dfstat['statreg']=='North Hollywood']['transit_score'].mean()
dfstat.isna().sum()
dfstat['statreg'].unique()
plt.figure(figsize = (6,7))
plt.scatter(dfstat[dfstat['statreg']=='DTLA']['statlong'], dfstat[dfstat['statreg']=='DTLA']['statlat'],alpha=0.4)
plt.scatter(dfstat[dfstat['statreg']=='Port of LA']['statlong'], dfstat[dfstat['statreg']=='Port of LA']['statlat'],alpha=0.4)
plt.scatter(dfstat[dfstat['statreg']=='Westside']['statlong'], dfstat[dfstat['statreg']=='Westside']['statlat'],alpha=0.4)
plt.scatter(dfstat[dfstat['statreg']=='North Hollywood']['statlong'], dfstat[dfstat['statreg']=='North Hollywood']['statlat'],alpha=0.4)

plt.annotate('Port of LA', (dfstat[dfstat['statreg']=='Port of LA']['statlong'].mean(), dfstat[dfstat['statreg']=='Port of LA']['statlat'].mean()))
plt.annotate('Downtown Los Angeles', (dfstat[dfstat['statreg']=='DTLA']['statlong'].mean()-0.06, dfstat[dfstat['statreg']=='DTLA']['statlat'].mean()))
plt.annotate('Westside', (dfstat[dfstat['statreg']=='Westside']['statlong'].mean(), dfstat[dfstat['statreg']=='Westside']['statlat'].mean()))
plt.annotate('North Hollywood', (dfstat[dfstat['statreg']=='North Hollywood']['statlong'].mean(), dfstat[dfstat['statreg']=='North Hollywood']['statlat'].mean()))
plt.xlabel('Longtidude')
plt.ylabel('Latitude')
plt.grid(True)
print ('From the total of ', len(df), ' travels, there are only ', len(df[df['reg_start']!=df['reg_end']]), ' travels inter-region (', round(len(df[df['reg_start']!=df['reg_end']])/len(df)*100, 2), '%)')
center_pola_lon = dfstat[dfstat['statreg']=='Port of LA']['statlong'].mean()
center_pola_lat = dfstat[dfstat['statreg']=='Port of LA']['statlat'].mean()
center_dtla_lon = dfstat[dfstat['statreg']=='DTLA']['statlong'].mean()
center_dtla_lat = dfstat[dfstat['statreg']=='DTLA']['statlat'].mean()
center_west_lon = dfstat[dfstat['statreg']=='Westside']['statlong'].mean()
center_west_lat = dfstat[dfstat['statreg']=='Westside']['statlat'].mean()
center_noh_lon = dfstat[dfstat['statreg']=='North Hollywood']['statlong'].mean()
center_noh_lat = dfstat[dfstat['statreg']=='North Hollywood']['statlat'].mean()
dist_c = []
for i in range(len(dfstat)): 
    if dfstat['statreg'].iloc[i] == 'Port of LA':
        dist_c.append(round(np.sqrt(((dfstat.statlong.iloc[i] - center_pola_lon)**2) + ((dfstat.statlat.iloc[i] - center_pola_lat)**2)), 4))
    elif dfstat['statreg'].iloc[i] == 'DTLA':
        dist_c.append(round(np.sqrt(((dfstat.statlong.iloc[i] - center_dtla_lon)**2) + ((dfstat.statlat.iloc[i] - center_dtla_lat)**2)), 4))
    elif dfstat['statreg'].iloc[i] == 'Westside':
        dist_c.append(round(np.sqrt(((dfstat.statlong.iloc[i] - center_west_lon)**2) + ((dfstat.statlat.iloc[i] - center_west_lat)**2)), 4))
    elif dfstat['statreg'].iloc[i] == 'North Hollywood':
        dist_c.append(round(np.sqrt(((dfstat.statlong.iloc[i] - center_noh_lon)**2) + ((dfstat.statlat.iloc[i] - center_noh_lat)**2)), 4))
        
dfstat['dist_creg'] = dist_c
dfcenter = pd.DataFrame([
    {"region": 'DTLA',
    "lat":center_dtla_lat,
    "long": center_dtla_lon},
    {"region": 'Port of LA',
    "lat":center_pola_lat,
    "long": center_pola_lon},
    {"region": 'Westside',
    "lat":center_west_lat,
    "long": center_west_lon},
    {"region": 'North Hollywood',
    "lat":center_noh_lat,
    "long": center_noh_lon}])

dfcenter.to_csv('dfcenter.csv')
df['date'] = df['start_time'].apply(lambda x: x.date())

def weekend(x):
    if x.weekday() == 5 or x.weekday() == 6:
        return 1
    else:
        return 0

df['weekend'] = df['date'].apply(weekend)
df['hour'] = df['start_time'].apply(lambda x: x.hour)
df['temp_avg'] = (df['tempmax'] + df['tempmin'])/2
df['temp_avg'].describe()
def temp(x):
    if x <= (df['temp_avg'].mean() - df['temp_avg'].std()):
        return 'cool'
    elif x > (df['temp_avg'].mean() - df['temp_avg'].std()) and x <= (df['temp_avg'].mean() + df['temp_avg'].std()):
        return 'regular'
    elif x > (df['temp_avg'].mean() + df['temp_avg'].std()):
        return 'warm'
    
df['temp_class'] = df['temp_avg'].apply(temp) 
encodertemp = OrdinalEncoder(categories=[['cool', 'regular', 'warm']])

df['temp_class'] = encodertemp.fit_transform(df['temp_class'].values.reshape(-1, 1))
def rain (x):
    if x >= 1:
        return 1
    else:
        return 0

df['rain'] = df['pcp'].apply(rain)
df.describe()
locs = dfstat[['statlat', 'statlong']]
locslist = locs.values.tolist()

# # setting the centre of the map using the mean of the lattitude and longitude of the stations
lat = ((dfstat[['statlat']].max() + dfstat[['statlat']].min())/2)+0.05
lon = ((dfstat[['statlong']].max() + dfstat[['statlong']].min())/2)+0.1

# making the map
map1 = folium.Map(location=[lat, lon], control_scale=True, zoom_start=[10.15])

# making the station markers (different colors for different region)
for point in range(0, len(locslist)):
    pop  = dfstat['name'][point] , dfstat['statreg'][point]
    if dfstat['statreg'][point] == 'DTLA':
        folium.Marker(locslist[point], popup=pop, icon=folium.Icon(color='cadetblue')).add_to(map1)
    elif dfstat['statreg'][point] == 'Westside':
        folium.Marker(locslist[point], popup=pop, icon=folium.Icon(color='orange')).add_to(map1)
    elif dfstat['statreg'][point] == 'Port of LA':
        folium.Marker(locslist[point], popup=pop, icon=folium.Icon(color='lightblue')).add_to(map1)
    elif dfstat['statreg'][point] == 'North Hollywood':
        folium.Marker(locslist[point], popup=pop, icon=folium.Icon(color='darkpurple')).add_to(map1)
map1
x = df.groupby(['start_station', 'end_station', 'reg_start', 'reg_end'], as_index=False).count().sort_values(by='trip_id', ascending=False).head(10)
x[['start_station', 'end_station', 'reg_start', 'reg_end', 'trip_id']]
# extracting the latitude and longitude point of the stations
xy3030 = [float(dfstat[dfstat['statid']==3030]['statlat']), float(dfstat[dfstat['statid']==3030]['statlong'])]
xy3014 = [float(dfstat[dfstat['statid']==3014]['statlat']), float(dfstat[dfstat['statid']==3014]['statlong'])]

# setting the centre of the map using the mean of the lattitude and longitude of the stations
lat2 = (xy3030[0]+xy3014[0])/2
lon2 = (xy3030[1]+xy3014[1])/2

# # making the map
map2 = folium.Map(location=[lat2, lon2], control_scale=True, zoom_start=[15.7])
folium.Marker(xy3030, popup=dfstat[dfstat['statid']==3030]['name'].iloc[0], icon=folium.Icon(color='orange')).add_to(map2)
folium.Marker(xy3014, popup=dfstat[dfstat['statid']==3014]['name'].iloc[0], icon=folium.Icon(color='red')).add_to(map2)

map2
df['duration'].describe()
plt.figure(figsize = (22,3))
plt.boxplot(df['duration'], vert=False)
plt.xticks(range(0, 1441, 40))
plt.grid(True)
iqr = np.subtract(*np.percentile(df['duration'], [75, 25]))
print('IQR = ', iqr)
upper = np.percentile(df['duration'], 75) + (1.5 * iqr)
print('upper = ', (np.percentile(df['duration'], 75)) + (1.5 * iqr))
print ('numbers of outlier: ', len(df[df['duration']> upper]),', is', round(len(df[df['duration']>upper])/len(df)*100, 2),'% from the total data')

df['plan_duration'].unique()
df[df['plan_duration']==999]
df = df[df['plan_duration']!=999]
plt.pie([len(df[df['plan_duration'] == 1]), len(df[df['plan_duration'] == 30]), len(df[df['plan_duration'] == 365])],labels= ['One Day', 'One Month', 'One Year'], autopct='%2f%%')
plt.show()
df['trip_route_category'].unique()
plt.pie([len(df[df['trip_route_category'] == 'One Way']), len(df[df['trip_route_category'] == 'Round Trip'])],labels= ['One Way', 'Round Trip'], autopct='%2f%%')
plt.show()
df['passholder_type'].unique()
plt.pie([len(df[df['passholder_type'] =='Monthly Pass']), len(df[df['passholder_type'] =='Walk-up']), len(df[df['passholder_type'] =='One Day Pass']),
        len(df[df['passholder_type'] =='Annual Pass']), len(df[df['passholder_type'] =='Flex Pass'])],
       labels= ['Monthly Pass', 'Walk-up', 'One Day Pass', 'Annual Pass', 'Flex Pass'], autopct='%2f%%')
plt.show()
weekpass = df.groupby(['weekend', 'passholder_type'], as_index=False).count()

plt.figure(figsize=(12,10))
plt.subplot(121)
plt.pie(weekpass[weekpass['weekend']==1]['trip_id'], labels =['Annual Pass', 'Flex Pass', 'Monthly Pass', 'One Day Pass', 'Walk Up'], autopct='%2f')
plt.title('Passholder Type Weekend Trip')

plt.subplot(122)
plt.pie(weekpass[weekpass['weekend']==0]['trip_id'], labels =['Annual Pass', 'Flex Pass', 'Monthly Pass', 'One Day Pass', 'Walk Up'], autopct='%2f')
plt.title('Passholder Type Weekday Trip')
plt.show()
df2 = df.drop('plan_duration', axis = 1)
df['bike_type'].unique()
plt.pie([len(df[df['bike_type'] =='standard']), len(df[df['bike_type'] =='smart']), len(df[df['bike_type'] =='electric'])],
       labels= ['standard', 'smart', 'electric'], autopct='%2f%%')
plt.show()
# Grouping the trip based on hour and weekend

trip_hour = df.groupby(['time'])['trip_id'].count()
west_end = df.groupby(['reg_start', 'weekend', 'time'])['trip_id'].count().loc['Westside'].loc[1]
west_day = df.groupby(['reg_start', 'weekend', 'time'])['trip_id'].count().loc['Westside'].loc[0]

pola_end = df.groupby(['reg_start', 'weekend', 'time'])['trip_id'].count().loc['Port of LA'].loc[1]
pola_day = df.groupby(['reg_start', 'weekend', 'time'])['trip_id'].count().loc['Port of LA'].loc[0]

dtla_end = df.groupby(['reg_start', 'weekend', 'time'])['trip_id'].count().loc['DTLA'].loc[1]
dtla_day = df.groupby(['reg_start', 'weekend', 'time'])['trip_id'].count().loc['DTLA'].loc[0]

nh_end = df.groupby(['reg_start', 'weekend', 'time'])['trip_id'].count().loc['North Hollywood'].loc[1]
nh_day = df.groupby(['reg_start', 'weekend', 'time'])['trip_id'].count().loc['North Hollywood'].loc[0]

# filling in hours without rides with 0 so that it can be plotted
west_end.loc[3] = 0
pola_end.loc[3] = 0; pola_end.loc[5] = 0; pola_end.loc[4] = 0; pola_end.loc[6] = 0; pola_end.loc[0] = 0;
nh_end.loc[5] = 0; nh_end.loc[4] = 0; nh_end.loc[7] = 0; nh_day.loc[3] = 0;
pola_day.loc[0] = 0; pola_day.loc[1] = 0; pola_day.loc[2] = 0;

# plotting the weekday trips

plt.figure(figsize = (15,7))
plt.plot(range(0,24), west_day)
plt.plot(range(0,24), pola_day)
plt.plot(range(0,24), dtla_day) 
plt.plot(range(0,24), nh_day) 
plt.xticks(range(0,24))
plt.grid(True)
plt.title('Trips per Hour on Weekday')
plt.legend(['Westside', 'Port of LA', 'DTLA',  'North Hollywood'])

# plotting the weekend trips

plt.figure(figsize = (15,7))
plt.plot(range(0,24), west_end)
plt.plot(range(0,24), pola_end)
plt.plot(range(0,24), dtla_end) 
plt.plot(range(0,24), nh_end) 
plt.xticks(range(0,24))
plt.grid(True)
plt.title('Trips per Hour on Weekend')
plt.legend(['Westside', 'Port of LA', 'DTLA',  'North Hollywood'])

df['reg_start'].unique()
region_start = pd.get_dummies(df['reg_start']) 
df = pd.concat([df, region_start[['Port of LA', 'DTLA', 'North Hollywood']]], axis = 1)
plt.pie([len(df[df['reg_start'] =='DTLA']), len(df[df['reg_start'] =='Port of LA']), len(df[df['reg_start'] =='Westside']), len(df[df['reg_start'] =='North Hollywood'])],
       labels= ['DTLA', 'Port of LA', 'Westside', 'North Hollywood'], autopct='%2f%%')
plt.show()
plt.pie([len(df[df['temp_class']==0]), len(df[df['temp_class']==1]), len(df[df['temp_class']==2])], autopct='%2f%%',  labels= ['cooler', 'regular', 'warmer'])
plt.show()
plt.pie([len(df[df['rain']==1]), len(df[df['rain']==0])], autopct='%2f', explode=[0,0.7], labels= ['rain', 'clear'])
plt.show()
plt.pie([len(df[df['weekend']==1]), len(df[df['weekend']==0])], autopct='%2f')
plt.show()
dfstat['walk_score'].describe()
plt.figure(figsize = (16,6))
plt.subplot(121)
plt.hist(dfstat['walk_score'], bins =100)
plt.subplot(122)
plt.hist(dfstat[dfstat['statreg']=='DTLA']['walk_score'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='Port of LA']['walk_score'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='North Hollywood']['walk_score'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='Westside']['walk_score'], bins =50, alpha=0.6)
plt.legend(['DTLA', 'Port of LA', 'North Hollywood', 'Westside'])
plt.show()
dfstat.groupby('statreg').describe()['walk_score']
dfstat['bike_score'].describe()
plt.figure(figsize = (16,6))
plt.subplot(121)
plt.hist(dfstat['bike_score'], bins =100)
plt.subplot(122)
plt.hist(dfstat[dfstat['statreg']=='DTLA']['bike_score'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='Port of LA']['bike_score'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='North Hollywood']['bike_score'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='Westside']['bike_score'], bins =50, alpha=0.6)
plt.legend(['DTLA', 'Port of LA', 'North Hollywood', 'Westside'])
plt.show()
dfstat.groupby('statreg').describe()['bike_score']
dfstat['transit_score'].describe()
plt.figure(figsize = (16,6))
plt.subplot(121)
plt.hist(dfstat['transit_score'], bins =100)
plt.subplot(122)
plt.hist(dfstat[dfstat['statreg']=='DTLA']['transit_score'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='Port of LA']['transit_score'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='North Hollywood']['transit_score'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='Westside']['transit_score'], bins =50, alpha=0.6)
plt.legend(['DTLA', 'Port of LA', 'North Hollywood', 'Westside'])
plt.show()
dfstat.groupby('statreg').describe()['transit_score']
dfstat['dist_creg'].describe()
plt.figure(figsize = (16,6))
plt.subplot(121)
plt.hist(dfstat['dist_creg'], bins =100)
plt.subplot(122)
plt.hist(dfstat[dfstat['statreg']=='Westside']['dist_creg'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='DTLA']['dist_creg'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='Port of LA']['dist_creg'], bins =50, alpha=0.6)
plt.hist(dfstat[dfstat['statreg']=='North Hollywood']['dist_creg'], bins =50, alpha=0.6)
plt.legend(['DTLA', 'Port of LA', 'North Hollywood', 'Westside'])
plt.show()
dfstat.groupby('statreg').describe()['dist_creg']
df.to_csv('df.csv')
dfstat.to_csv('dfstat.csv')
df.head()
dfx = (df[['trip_id', 'start_station', 'weekend', 'rain', 'temp_class', 'date',
       'North Hollywood', 'Port of LA', 'DTLA']]).groupby(['start_station', 'date', 'weekend', 'rain', 'temp_class',
       'North Hollywood', 'Port of LA', 'DTLA'], as_index=False).count()
dfx = dfx.rename(columns={'trip_id':'total_trip', 'start_station':'station_id'})
dfx
dfx.head()
dfx['stat_name'] = dfx['station_id'].apply(lambda x: dfstat[dfstat['statid']==x]['name'].iloc[0])
dfx['lat'] = dfx['station_id'].apply(lambda x: dfstat[dfstat['statid']==x]['statlat'].iloc[0])
dfx['long'] = dfx['station_id'].apply(lambda x: dfstat[dfstat['statid']==x]['statlong'].iloc[0])
dfx['walk_score'] = dfx['station_id'].apply(lambda x: dfstat[dfstat['statid']==x]['walk_score'].iloc[0])
dfx['transit_score'] = dfx['station_id'].apply(lambda x: dfstat[dfstat['statid']==x]['transit_score'].iloc[0])
dfx['bike_score'] = dfx['station_id'].apply(lambda x: dfstat[dfstat['statid']==x]['bike_score'].iloc[0])
dfx['dregioncenter'] = dfx['station_id'].apply(lambda x: dfstat[dfstat['statid']==x]['dist_creg'].iloc[0])
dfx.head(5)
plt.figure(figsize = (14,8))
plt.subplot(211)
plt.hist(dfx['total_trip'], bins=200)
plt.title ('Distribution of Trip per Station')
plt.subplot(212)
plt.boxplot(dfx['total_trip'], vert=False)
plt.title ('Distribution of Trip per Station')
plt.show()
dfx['total_trip'].describe()
transformerlog = FunctionTransformer(np.log10, validate=True)
dfx['trip_log'] = transformerlog.transform(dfx[['total_trip']])
plt.figure(figsize = (14,4))
plt.hist(dfx['trip_log'], bins=200)
plt.title ('Log Transformation of Trip Distribution per Day')
plt.show()
dfx.to_csv('dfx.csv')
dfx
data = dfx
data['trip_ori'] = data['total_trip'] 
data=data.drop(['total_trip'], axis =1)
plt.figure(figsize = (16,10))
sb.heatmap(data.corr(), annot = True, cmap="PiYG")
df = data[['trip_log', 'trip_ori', 'rain', 'weekend' , 'temp_class', 'transit_score', 'bike_score', 'dregioncenter',  'walk_score']]
df.head(3)
plt.figure(figsize = (14,7))

plt.subplot(221)
plt.scatter(data['walk_score'], data['trip_log'])
plt.title('walk score vs trip log')

plt.subplot(222)
plt.scatter(data['transit_score'], data['trip_log'])
plt.title('bike score vs trip log')

plt.subplot(223)
plt.scatter(data['bike_score'], data['trip_log'])
plt.title('bike score vs trip log')

plt.subplot(224)
plt.scatter(data['dregioncenter'], data['trip_log'])
plt.title('distance to region center vs trip log')
df['trip_log'].describe()
scaler1 = MinMaxScaler()
scaler1.fit(df[['temp_class', 'transit_score', 'bike_score', 'dregioncenter',  'walk_score']])
scaledx = scaler1.transform(df[['temp_class', 'transit_score', 'bike_score', 'dregioncenter',  'walk_score']])
scaledx = pd.DataFrame(scaledx, columns = ['temp_class', 'transit_score', 'bike_score', 'dregioncenter',  'walk_score']) 
dfs = df.drop(['temp_class', 'transit_score', 'bike_score', 'dregioncenter',  'walk_score'], axis = 1)
dfs = pd.concat([dfs, scaledx], axis = 1)
xtr, xts, ytr, yts = train_test_split(dfs[['rain', 'weekend' , 'temp_class', 'transit_score', 'bike_score', 'dregioncenter',  'walk_score']],
                                     dfs[['trip_log', 'trip_ori']], test_size = 0.1)
def evalmet (model, x, y):
    ypred = model.predict(x)
    evalmetrics = pd.DataFrame()
    evalmetrics['RMSE'] = [np.sqrt(mean_squared_error(y, ypred))]
    evalmetrics['MAE'] = [mean_absolute_error(y, ypred)]
    evalmetrics['R2'] = [r2_score(y, ypred)]
    return evalmetrics
modellinreg = LinearRegression()
modellinreg.get_params()
paramdist_linreg = {
    'fit_intercept': [True, False],
    'normalize': [True, False]
}
modelrs_linreg = RandomizedSearchCV(
    estimator = modellinreg, param_distributions = paramdist_linreg, cv = 10, random_state=10)
modelrs_linreg.fit(xtr, ytr['trip_log'])
modelrs_linreg.best_params_
modellinreg2 = LinearRegression(normalize=False, fit_intercept=True)
modellinreg2.fit(xtr, ytr['trip_log'])
evhp_linreg2 = evalmet(modellinreg2, xtr, ytr['trip_log'])
evhp_linreg2
modelridge = Ridge()
modelridge.get_params()
paramdist_ridge = {
    'alpha': np.arange(0.1, 2, 0.05),
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    'max_iter': range(1000,2000, 100)
}
modelr = Ridge()
modelRgs = GridSearchCV(modelr, paramdist_ridge, cv=5)
modelRgs.fit(xtr, ytr['trip_log'])
modelRgs.best_params_
modelridge = Ridge()
modelridge.fit(xtr, ytr['trip_log'])
evhp_ridge = evalmet(modelridge, xtr, ytr['trip_log'])
evhp_ridge
modelridge2 = Ridge(solver = 'saga', max_iter = 1600, alpha=0.40000000000000013)
modelridge2.fit(xtr, ytr['trip_log'])
evhp_ridge2 = evalmet(modelridge2, xtr, ytr['trip_log'])
evhp_ridge2
rl = []
rr = []

rl.append(cross_val_score(
    LinearRegression(), xtr, 
    ytr['trip_log'], cv = 10, scoring ='r2').mean())
rl.append(cross_val_score(
    LinearRegression(), xtr, 
    ytr['trip_log'], cv = 10, scoring ='neg_root_mean_squared_error').mean())
rl.append (cross_val_score(
    LinearRegression(), xtr, 
    ytr['trip_log'], cv = 10, scoring ='neg_mean_absolute_error').mean())
                                        
rr.append(cross_val_score(
    Ridge(solver = 'saga', max_iter = 1600, alpha=0.40000000000000013), xtr, 
    ytr['trip_log'], cv = 10, scoring ='r2').mean())
rr.append(cross_val_score(
    Ridge(solver = 'saga', max_iter = 1600, alpha=0.40000000000000013), xtr, 
    ytr['trip_log'], cv = 10, scoring ='neg_root_mean_squared_error').mean())
rr.append (cross_val_score(
    Ridge(solver = 'saga', max_iter = 1600, alpha=0.40000000000000013), xtr, 
    ytr['trip_log'], cv = 10, scoring ='neg_mean_absolute_error').mean())
dfcv1 = pd.DataFrame(['R2', 'RMSE', 'MAE'], columns = ['Evaluation Score'])
dfcv1['LinearRegression'] = rl
dfcv1['Ridge'] = rr
dfcv1['selisih'] = dfcv1['LinearRegression'] - dfcv1['Ridge'] 

dfcv1
model_lin = LinearRegression()
model_lin.fit(xtr, ytr['trip_log'])
evaltrain1 = evalmet (model_lin, xtr, ytr['trip_log'])
yts2l = yts
yts2l['ypred_log'] = model_lin.predict(xts)
evaltes2l = evalmet (model_lin, xts, yts['trip_log'])
df_eval1 = pd.concat([evaltrain1, evaltes2l], axis = 0)
df_eval1['data'] = ['train', 'test']
df_eval1
yts2l['residual'] = yts2l['trip_log'] - yts2l['ypred_log']

plt.figure(figsize = (10,10))
plt.subplot(221)
plt.scatter(yts2l['trip_log'], yts2l['residual'])
plt.grid(True)
plt.title('Prediction vs Residual')
plt.subplot(222)
plt.scatter(yts2l['ypred_log'], yts2l['residual'])
plt.grid(True)
plt.title('Observed Value vs Residual')
plt.figure(figsize = (14,8))
sb.heatmap(df.corr("spearman"), annot = True, cmap="PiYG")
xtr, xts, ytr, yts = train_test_split(df[['rain', 'weekend' , 'temp_class', 'transit_score', 'bike_score', 'dregioncenter',  'walk_score']],
                                     df[['trip_log', 'trip_ori']], test_size = 0.1)
modelrfr = RandomForestRegressor()
modelrfr.get_params()
paramdist_rfr = {
    'min_samples_split': range(2,10),
    'n_estimators': range(10,200, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}
modelrs_rfr = RandomizedSearchCV(
    estimator = modelrfr, param_distributions = paramdist_rfr, cv = 10, random_state=11)
modelrs_rfr.fit(xtr, ytr['trip_log'])
modelrs_rfr.best_params_
modelrfr = RandomForestRegressor()
modelrfr.fit(xtr, ytr['trip_log'])
evhp_rfr = evalmet(modelrfr, xtr, ytr['trip_log'])
evhp_rfr
modelrfr2 = RandomForestRegressor(n_estimators=140, min_samples_split=8, max_features='sqrt')
modelrfr2.fit(xtr, ytr['trip_log'])
evhp_rfr2 = evalmet(modelrfr2, xtr, ytr['trip_log'])
evhp_rfr2
modelgbr = GradientBoostingRegressor()
modelgbr.get_params()
paramdist_gbr = {
    'n_estimators': range(100,1000, 50),
    'learning_rate': np.arange(0.01, 1, 0.05),
    'min_samples_split': range (2,10)
}
modelrs_gbr = RandomizedSearchCV(
    estimator = modelgbr, param_distributions = paramdist_gbr, cv = 10, random_state=13)
modelrs_gbr.fit(xtr, ytr['trip_log'])
modelrs_gbr.best_params_
modelgbr = GradientBoostingRegressor()
modelgbr.fit(xtr, ytr['trip_log'])
evhp_gbr = evalmet(modelgbr, xtr, ytr['trip_log'])
evhp_gbr
modelgbr2 = GradientBoostingRegressor(n_estimators = 600, learning_rate=0.26, min_samples_split= 3)
modelgbr2.fit(xtr, ytr['trip_log'])
evhp_gbr2 = evalmet(modelgbr2, xtr, ytr['trip_log'])
evhp_gbr2
rfr = []
gbr = []

rfr.append(cross_val_score(
    RandomForestRegressor(), xtr, 
    ytr['trip_log'], cv = 10, scoring ='r2').mean())
rfr.append(cross_val_score(
    RandomForestRegressor(), xtr, 
    ytr['trip_log'], cv = 10, scoring ='neg_root_mean_squared_error').mean())
rfr.append (cross_val_score(
    RandomForestRegressor(), xtr, 
    ytr['trip_log'], cv = 10, scoring ='neg_mean_absolute_error').mean())
                                        
gbr.append(cross_val_score(
    GradientBoostingRegressor(n_estimators = 600, learning_rate=0.26, min_samples_split= 3), xtr, 
    ytr['trip_log'], cv = 10, scoring ='r2').mean())
gbr.append(cross_val_score(
    GradientBoostingRegressor(n_estimators = 600, learning_rate=0.26, min_samples_split= 3), xtr, 
    ytr['trip_log'], cv = 10, scoring ='neg_root_mean_squared_error').mean())
gbr.append (cross_val_score(
   GradientBoostingRegressor(n_estimators = 600, learning_rate=0.26, min_samples_split= 3), xtr, 
    ytr['trip_log'], cv = 10, scoring ='neg_mean_absolute_error').mean())

dfcv = pd.DataFrame(['R2', 'RMSE', 'MAE'], columns = ['Evaluation Score'])
dfcv['RandomForestRegressor'] = rfr
dfcv['GradientBoostingRegressor'] = gbr

dfcv
model_best = GradientBoostingRegressor(n_estimators = 600, learning_rate=0.26, min_samples_split= 3)
model_best.fit(xtr, ytr['trip_log'])
ytr['ypred_log'] = model_best.predict(xtr)
ytr['ypred_ori'] = ytr['ypred_log'].apply(lambda x: 10**x)
ytr.head()
evaltrain = evalmet (model_best, xtr, ytr['trip_log'])
evaltrain
ytsx = yts[['trip_log', 'trip_ori']]
ytsx['ypred_log'] = model_best.predict(xts)
ytsx.head()
evaltes = evalmet (model_best, xts, ytsx['trip_log'])
evaltes
df_eval = pd.concat([evaltrain, evaltes], axis = 0)
df_eval['data'] = ['train', 'test']
df_eval
feature_importance = model_best.feature_importances_
dfimportance = pd.DataFrame()
dfimportance['features'] = xtr.columns
dfimportance['importances'] = feature_importance
dfimportance = dfimportance.sort_values(by='importances', ascending=False)
dfimportance 