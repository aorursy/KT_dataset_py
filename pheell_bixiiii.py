import datetime

import pandas as pd

import numpy as np

import geopandas as gpd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

from sklearn.preprocessing import StandardScaler



# Read Montreal map and station info

arrondissements = gpd.read_file('../input/montreal.geojson')

stations = pd.read_csv('../input/Stations_2018.csv')

station_info = pd.read_json('../input/bixi_station_information.json')

station_info = pd.DataFrame(station_info['data'][0])[['capacity','short_name']]

station_info = station_info[station_info.short_name != 'Demo_BIXI']

station_info['code'] = station_info['short_name'].astype('int64')

station_info = station_info.drop('short_name', axis=1)



# Read trips info

OD_files = ['../input/OD_2018-' + n + '.csv' for n in ['04', '05', '06', '07', '08', '09', '10', '11']]

trips = pd.concat((pd.read_csv(f) for f in OD_files)).reset_index(drop=True)

trips['start_date'] = trips['start_date'].astype('datetime64')

trips['end_date'] = trips['end_date'].astype('datetime64')

trips = trips[trips.start_station_code != trips.end_station_code] # drop round trips

del OD_files

print(trips.shape[0], 'trips') # 5,181,510 trips



# Get in/out counts from trips

in_out = pd.concat([trips['start_station_code'].value_counts(), trips['end_station_code'].value_counts()], axis=1).reset_index()

in_out.columns = ['code', 'out', 'in']



# Merge into stations df

stations = stations.merge(station_info, on='code')

stations = stations.merge(in_out, on='code')

del station_info

del in_out
# Rank stations based on outs, ins

print(stations.sort_values(['out'], ascending=False)[['name','out']])

print(stations.sort_values(['in'], ascending=False)[['name','in']])



# stations.difference = ins - outs per dock per day

total_days = (datetime.datetime(2018,11,16) - datetime.datetime(2018,4,10)).days

stations['difference'] = (stations['in'] - stations['out'])/stations['capacity']/total_days

print(stations.sort_values(['difference'], ascending=False)[['name','difference']])
# Plot it; clive along Sherbrooke??



shiftmax = stations.difference.quantile(0.98)

shiftmin = stations.difference.quantile(0.02)

shiftmid = 1 - shiftmax / (shiftmax + abs(shiftmin))



cdict = {'red':   [(0, 0, 0), (shiftmid, 1, 1), (1, 1, 1)],

         'green': [(0, 0, 0), (shiftmid, 1, 1), (1, 0, 0)],

         'blue':  [(0, 1, 1), (shiftmid, 1, 1), (1, 0, 0)]}

cmap = matplotlib.colors.LinearSegmentedColormap('shiftedzero', segmentdata=cdict, N=256)



arrondissements.plot(facecolor='grey', edgecolor='white', linewidth=.5)

plt.scatter(stations.longitude, stations.latitude, cmap=cmap, alpha=.5, c=stations.difference, s=stations.capacity)

plt.xlim(-73.67, -73.48); plt.ylim(45.4, 45.59)

plt.colorbar(); plt.axis('off')

plt.text(-73.65, 45.4,'Extra bikes left per dock per day', fontsize=12)

plt.clim(shiftmin, shiftmax)

plt.rcParams["figure.figsize"] = (12, 6)
# Define timewindows and working/off days

holidays = [datetime.datetime(2018,5,21), datetime.datetime(2018,6,25), datetime.datetime(2018,7,2), datetime.datetime(2018,9,3), datetime.datetime(2018,10,8)]

working_days = np.busday_count(np.datetime64('2018-04-10'), np.datetime64('2018-11-16')) - len(holidays)

off_days = total_days - working_days

timewindows = [(0, 6), (6, 10), (10, 15), (15, 19), (19, 24)]

timewindow_labels = [str(start) + '-' + str(end) for start, end in timewindows]



def cat_time(time):

    for start, end in timewindows: 

        if time.hour in range(start, end): return str(start) + '-' + str(end)

def cat_workday(date):

    if date.weekday() < 5 and date not in holidays: return 'wk'

    else: return 'wkend'



trips['workday'] = trips['start_date'].apply(cat_workday)

trips['timewindow'] = trips['start_date'].apply(cat_time)



# Generate 10 out columns (5 timewindows * {wk, wkend})

outs = trips.groupby(['start_station_code','timewindow','workday']).count().drop(['end_station_code', 'end_date', 'duration_sec', 'is_member'], axis = 1).reset_index()

outs.columns = ['code', 'timewindow', 'workday', 'count']

outs = outs.pivot_table(columns=['timewindow','workday'], index='code', values='count', fill_value=0)

outs.columns = ['out_' + '_'.join(col).strip() for col in outs.columns.values]

outs = outs.reset_index()



# Generate 10 in columns (5 timewindows * {wk, wkend})

ins = trips.groupby(['end_station_code','timewindow','workday']).count().drop(['start_station_code', 'start_date', 'duration_sec', 'is_member'], axis = 1).reset_index()

ins.columns = ['code', 'timewindow', 'workday', 'count']

ins = ins.pivot_table(columns=['timewindow','workday'], index='code', values='count', fill_value=0)

ins.columns = ['in_' + '_'.join(col).strip() for col in ins.columns.values]

ins = ins.reset_index()



# merge into stations df

stations = stations.merge(outs, on='code')

stations = stations.merge(ins, on='code')

del ins; del outs
# Plot raw counts



df = stations

#variable = 'in_6-10_wk' #work hubs (mostly Downtown)

#variable = 'out_6-10_wk' #mostly Plateau

variable = 'out_0-6_wkend' #mostly along the Main and orange line

cmap = matplotlib.cm.get_cmap('plasma_r')



print(df.sort_values([variable], ascending=False)[['name',variable]])



arrondissements.plot(facecolor='grey', edgecolor='white', linewidth=.5)

plt.scatter(stations.longitude, stations.latitude, cmap=cmap, alpha=.5, c=df[variable], s=stations.capacity)

plt.xlim(-73.67, -73.48); plt.ylim(45.4, 45.59)

plt.colorbar(); plt.axis('off')

plt.rcParams["figure.figsize"] = (12, 6)
# Define share of ins/outs of each time window 

# Define difference (ins - outs per dock per day) for each time window

for timewindow in timewindow_labels:

    stations['propor_in_' + timewindow + '_wk'] = stations['in_' + timewindow + '_wk'] / (np.sum(stations['in_' + timewindow + '_wk'] for timewindow in timewindow_labels) + np.sum(stations['in_' + timewindow + '_wkend'] for timewindow in timewindow_labels))

    stations['propor_out_' + timewindow + '_wk'] = stations['out_' + timewindow + '_wk'] / (np.sum(stations['out_' + timewindow + '_wk'] for timewindow in timewindow_labels) + np.sum(stations['out_' + timewindow + '_wkend'] for timewindow in timewindow_labels))

    stations['propor_in_' + timewindow + '_wkend'] = stations['in_' + timewindow + '_wkend'] / (np.sum(stations['in_' + timewindow + '_wkend'] for timewindow in timewindow_labels) + np.sum(stations['in_' + timewindow + '_wk'] for timewindow in timewindow_labels))

    stations['propor_out_' + timewindow + '_wkend'] = stations['out_' + timewindow + '_wkend'] / (np.sum(stations['out_' + timewindow + '_wkend'] for timewindow in timewindow_labels) + np.sum(stations['out_' + timewindow + '_wk'] for timewindow in timewindow_labels))

    stations['diff_' + timewindow + '_wk'] = (stations['in_' + timewindow + '_wk'] - stations['out_' + timewindow + '_wk'])/stations['capacity']/working_days

    stations['diff_' + timewindow + '_wkend'] = (stations['in_' + timewindow + '_wkend'] - stations['out_' + timewindow + '_wkend'])/stations['capacity']/off_days



# Rank stations by 6-10 week difference, share of 6-10 week ins and outs

print(stations.sort_values(['diff_6-10_wk'], ascending=False)[['name','diff_6-10_wk']])

print(stations.sort_values(['propor_in_6-10_wk'], ascending=False)[['name','propor_in_6-10_wk']])

print(stations.sort_values(['propor_out_6-10_wk'], ascending=False)[['name','propor_out_6-10_wk']])
# Define diff_in/out, wk(end)_diff

stations['wkend_out'] = np.sum(stations['out_' + timewindow + '_wkend'] for timewindow in timewindow_labels)/stations['capacity']/off_days

stations['wkend_in'] = np.sum(stations['in_' + timewindow + '_wkend'] for timewindow in timewindow_labels)/stations['capacity']/off_days

stations['wk_out'] = np.sum(stations['out_' + timewindow + '_wk'] for timewindow in timewindow_labels)/stations['capacity']/working_days

stations['wk_in'] = np.sum(stations['in_' + timewindow + '_wk'] for timewindow in timewindow_labels)/stations['capacity']/working_days

stations['diff_in'] = (stations['wk_in'] - stations['wkend_in'])

stations['diff_out'] = (stations['wk_out'] - stations['wkend_out'])

stations['wk_diff'] = (stations['wk_in'] - stations['wk_out'])

stations['wkend_diff'] = (stations['wkend_in'] - stations['wkend_out'])

stations['wk_wkend_diff'] = (stations['wk_diff'] + stations['wkend_diff'])/2

stations['diff_in_out'] = (stations['diff_in'] + stations['diff_out'])/2



# Rank stations by diff_in_out: measure of how much more station is used in the week vs weekend

# High: Sq Victoria, low: Atwater market, botanical garden etc.

print(stations.sort_values(['diff_in_out'])[['name','diff_in_out']])
"""Plot raw proportions"""

df = stations

variable = 'propor_in_6-10_wk' #work hubs



cmap = matplotlib.cm.get_cmap('plasma_r')



arrondissements.plot(facecolor='grey', edgecolor='white', linewidth=.5)

plt.scatter(stations.longitude, stations.latitude, cmap=cmap, alpha=.5, c=df[variable], s=stations.capacity)

plt.xlim(-73.67, -73.48); plt.ylim(45.4, 45.59)

plt.colorbar(); plt.axis('off')

plt.rcParams["figure.figsize"] = (12, 6)
""" Scale all numeric columns"""

nonnumeric_columns = ['code', 'name', 'latitude', 'longitude', 'capacity']

numeric_columns = [column for column in stations.columns if column not in nonnumeric_columns]

stations_scaled = stations[numeric_columns]

scaler = StandardScaler().fit(stations_scaled.values)

stations_scaled = scaler.transform(stations_scaled.values)

stations_scaled = pd.concat([stations[nonnumeric_columns], pd.DataFrame(stations_scaled, columns=numeric_columns)], axis=1)
"""Plot scaled data"""



df = stations_scaled

variable = 'wk_wkend_diff' #pretty much original difference plot

#variable = 'diff_in_out' # how much more station is used in the week vs holiday



#variable = 'propor_in_6-10_wk'

#variable = 'propor_out_6-10_wk'

#variable = 'diff_6-10_wk'



#variable = 'propor_in_15-19_wk'

#variable = 'propor_out_15-19_wk'

#variable = 'diff_15-19_wk'



#variable = 'propor_in_0-6_wkend'

#variable = 'propor_out_0-6_wkend'

#variable = 'diff_0-6_wkend'



shiftmax = df[variable].quantile(0.98)

shiftmin = df[variable].quantile(0.02)

shiftmid = 1 - shiftmax / (shiftmax + abs(shiftmin))



cdict = {'red':   [(0, 0, 0), (shiftmid, 1, 1), (1, 1, 1)],

         'green': [(0, 0, 0), (shiftmid, 1, 1), (1, 0, 0)],

         'blue':  [(0, 1, 1), (shiftmid, 1, 1), (1, 0, 0)]}

cmap = matplotlib.colors.LinearSegmentedColormap('shiftedzero', segmentdata=cdict, N=256)



arrondissements.plot(facecolor='grey', edgecolor='white', linewidth=.5)

plt.scatter(df.longitude, df.latitude, cmap=cmap, alpha=.5, c=df[variable], s=df.capacity)

plt.xlim(-73.67, -73.48); plt.ylim(45.4, 45.59)

plt.colorbar(); plt.axis('off')

plt.clim(shiftmin, shiftmax)

plt.rcParams["figure.figsize"] = (12, 6)
df = stations_scaled

var_x = 'propor_in_6-10_wk'

var_y = 'propor_out_6-10_wk'

sns.scatterplot(df[var_x], df[var_y])

sns.lineplot(np.unique(df[var_x]), np.poly1d(np.polyfit(df[var_x], df[var_y], 1))(np.unique(df[var_x])), color='red')

plt.rcParams["figure.figsize"] = (12, 6)
""" Load elevations from json file; else download from Google API""" 



def elevation(lat, lng):

    apikey = "apikey"

    url = "https://maps.googleapis.com/maps/api/elevation/json"

    response = requests.get(url+"?locations="+str(lat)+","+str(lng)+"&key="+apikey)

    try:

        results = response.json().get('results')

        if 0 < len(results):

            elevation = results[0].get('elevation')

            # ELEVATION

            return elevation

        else:

            print('HTTP GET Request failed.')

    except ValueError:

        print('JSON decode failed: '+str(response))

        

#import requests

#elevations = [elevation(stations.loc[i].latitude, stations.loc[i].longitude) for i in range(len(stations))]

#stations['elevation'] = elevations

#stations_scaled['elevation'] = elevations

#elevations = dict(zip(stations.code, stations.elevation))

        

import json

with open('../input/elevations.json', 'r') as f:

    elevations2 = json.load(f)

elevations = {}

for key in elevations2:

    elevations[int(key)] = elevations2[key]

del elevations2

stations['elevation'] = stations['code'].apply(lambda e: elevations[e])

stations_scaled['elevation'] = stations_scaled['code'].apply(lambda e: elevations[e])
# Compute elevation change for each trip, rounded to 1 m.

trips['elevation_change'] = trips['end_station_code'].apply(lambda e: elevations[e]) - trips['start_station_code'].apply(lambda e: elevations[e])

trips['elevation_change'] = trips['elevation_change'].apply(lambda e: round(e))
# Plot elevation of stations

cmap = matplotlib.cm.get_cmap('plasma_r')

arrondissements.plot(facecolor='grey', edgecolor='white', linewidth=.5)

plt.scatter(stations.longitude, stations.latitude, cmap=cmap, alpha=.5, c=stations.elevation, s=stations.capacity)

plt.xlim(-73.67, -73.48); plt.ylim(45.4, 45.59)

plt.colorbar(); plt.axis('off')

plt.rcParams["figure.figsize"] = (12, 6)
# Histogram of trip elevation change

sns.distplot(trips.elevation_change, bins=75)

plt.xlim(-75, 75)

plt.rcParams["figure.figsize"] = (12, 6)
# Histogram of trip duration

sns.distplot(trips.duration_sec, bins=75)

plt.xlim(0,3500)

plt.rcParams["figure.figsize"] = (12, 6)
""" Sanity check: trip duration as a function trip elevation change

 effect of one positive meter > effect of one negative meter"""

sns.lineplot(x='elevation_change', y='duration_sec', data=trips[(trips['elevation_change'].between(-20,20))])

plt.rcParams["figure.figsize"] = (12, 6)

"""Plot station measure by station elevation"""



df = stations_scaled

#variable = 'diff_in_out' # no correlation

#variable = 'propor_in_6-10_wk' # no correlation

variable = 'wk_wkend_diff' #extra bikes left per dock per day

print(np.poly1d(np.polyfit(df['elevation'], df[variable], 1)).r[0]) # root: 43.98 m



sns.lineplot(df['elevation'].apply(lambda e: round(e)), df[variable])

sns.lineplot([120, 0], [0, 0], linewidth=2)

sns.lineplot(np.unique(df['elevation']), np.poly1d(np.polyfit(df['elevation'], df[variable], 1))(np.unique(df['elevation'])))

plt.ylim(-2.5, 2.5)

plt.rcParams["figure.figsize"] = (12, 6)
from sklearn.cluster import KMeans



useful_columns = [col for col in numeric_columns if col != 'elevation']

X = stations_scaled[useful_columns].values

n_clusters = 2



km = KMeans(n_clusters=n_clusters).fit(X)



cmap = matplotlib.cm.get_cmap('plasma_r')

arrondissements.plot(facecolor='grey', edgecolor='white', linewidth=.5)

plt.scatter(stations.longitude, stations.latitude, cmap=cmap, alpha=.5, c=km.labels_, s=stations.capacity)

plt.xlim(-73.67, -73.48); plt.ylim(45.4, 45.59); plt.axis('off')

plt.rcParams["figure.figsize"] = (12, 6)
df = stations

cmap = matplotlib.cm.get_cmap('plasma_r')



fig, axs = plt.subplots(ncols = 3, subplot_kw={'xlim': (-73.67, -73.48), 'ylim': (45.4, 45.59)})

plt.rcParams["figure.figsize"] = (12, 6)

fig.set_facecolor('gray')



# Plot stations by elevation category --cutoff at 44m

axs[0].scatter(stations.longitude, stations.latitude, cmap=cmap, alpha=.5, c=stations['elevation']>=44.05, s=stations.capacity)

axs[0].axis('off')



# Plot stations by whether bike balance is positive or not

axs[1].scatter(stations.longitude, stations.latitude, cmap=cmap, alpha=.5, c=df['wk_wkend_diff']<=0.199, s=stations.capacity)

axs[1].axis('off')



# Clustering result

axs[2].scatter(stations.longitude, stations.latitude, cmap=cmap, alpha=.5, c=km.labels_, s=stations.capacity)

axs[2].axis('off')