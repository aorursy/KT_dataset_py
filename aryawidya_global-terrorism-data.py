# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cartopy.crs as ccrs

import cartopy





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_data = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv', engine='python')
pd.options.display.max_columns = 200

raw_data.head()
yearlyacts = raw_data.groupby('iyear',as_index = True, group_keys = False)['eventid'].count()

yearlyacts.plot(kind='line', figsize = (20,5),xticks = range(1970,2018),rot = 45, title='Yearly Cases of Terrorist Attacks',\

                linewidth = 3);





yearlyacts = raw_data.groupby(['iyear','region_txt'],as_index = False, group_keys = False)['eventid'].count()

#yearlyacts.region_txt.value_counts()

yearlyacts = yearlyacts.rename(columns={'region_txt': "Region", "eventid": "Cases"})



yearlyacts.pivot_table(index='iyear', columns='Region', aggfunc=np.sum, fill_value=0).plot(kind='line', figsize = (20,10),\

                                                                                              xticks = range(1970,2018),rot = 45,\

                                                                                              linewidth=5,title='Cases by Region');
atk_types = raw_data.groupby('attacktype1_txt', group_keys=False)['eventid'].count()

atk_types.nlargest(50).plot(kind = 'bar', figsize = (20,5),grid=True, rot=45, title='Types of Attack');
n_killsum = raw_data.groupby('country_txt',group_keys=False)['nkill'].sum()

n_cases = raw_data.groupby('country_txt', group_keys=False)['eventid'].count()

n_cases.nlargest(50).plot(kind = 'bar', figsize = (20,5),yticks = range(0,25001,2500),grid=True, title='Cases by Countries');
weapons = raw_data.groupby('weaptype1_txt', group_keys=False)['eventid'].count()

weapons.nlargest(50).plot(kind = 'bar', figsize = (20,5),grid=True, rot = 45, title= 'Weapon Used');

targets = raw_data.groupby('targtype1_txt', group_keys=False)['eventid'].count()

targets.nlargest(50).plot(kind = 'bar', figsize = (20,5),grid=True, rot=70, title='Targets');
print('Average number of perpetrators: ', round(raw_data[(raw_data.nperps > 0)]['nperps'].mean(),1))

print('Median of perpetrators: ', round(raw_data[(raw_data.nperps > 0)]['nperps'].median(),1))
#print(lon[lon < -180])

print(raw_data.at[17658,'longitude']) #change this wrong data



raw_data.at[17658,'longitude'] = -86.185896



print(raw_data.at[17658,'longitude'])
lat = raw_data[raw_data['iyear']>=2003]['latitude']

lon = raw_data[raw_data['iyear']>=2003]['longitude']





plt.figure(figsize = (20,15))



#extent = [-20, 40, 30, 60]

#central_lon = np.mean(extent[:2])

#central_lat = np.mean(extent[2:])



#ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))

ax = plt.axes(projection=ccrs.Mercator())



#ax.set_extent(extent,crs=ccrs.PlateCarree())

ax.scatter(lon,lat,transform=ccrs.PlateCarree(),s=1, c = 'red')

ax.coastlines()

ax.add_feature(cartopy.feature.BORDERS, linestyle="-")

ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)

plt.title('Worldwide Terrorists Attacks Since 2003', fontsize = 20);

df_filtered = raw_data[(raw_data['latitude']>25) & (raw_data['latitude']<60) & \

                (raw_data['longitude']> -20) & (raw_data['longitude']< 60) & (raw_data['INT_LOG']==0) & (raw_data['iyear']>=2003)]



lat2 = df_filtered['latitude']

lon2 = df_filtered['longitude']



data = pd.DataFrame({'latitude':lat2, 'longitude': lon2})

data = data.dropna(how="any")

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()

data_scaled = scaler.fit_transform(data)



db = DBSCAN(eps=0.3, min_samples=50).fit(data_scaled)



labels = db.labels_



data['cluster'] = labels

df_filtered['cluster'] = data['cluster']

# Number of clusters in labels, ignoring noise if present.

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

n_noise_ = list(labels).count(-1)



print('Estimated number of clusters: %d' % n_clusters_)

print('Estimated number of noise points: %d' % n_noise_)
plt.figure(figsize = (20,15))



#extent = [-20, 60, 25, 60]

#central_lon = np.mean(extent[:2])

#central_lat = np.mean(extent[2:])



legendname = []

for items in np.sort(df_filtered.cluster.unique()):

    if items == -1:

        legendname.append('Others / Cluster'+str(items))

    else:

        legendname.append('Cluster'+str(items))



#ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))

ax = plt.axes(projection=ccrs.Mercator())

#ax.set_extent(extent,crs=ccrs.PlateCarree())

scatter = ax.scatter(df_filtered.longitude,df_filtered.latitude,transform=ccrs.PlateCarree(), c = df_filtered.cluster,\

                     cmap = 'tab10', s = df_filtered['nkill']*10) #we define the marker size to corresponds to the number of casuality in the attack

ax.coastlines(linewidth = 2);

ax.add_feature(cartopy.feature.BORDERS, linestyle="-", linewidth = 2);

ax.add_feature(cartopy.feature.OCEAN);

#ax.add_feature(cartopy.feature.LAND, edgecolor='black')

ax.add_feature(cartopy.feature.LAKES, edgecolor='black');

ax.gridlines(crs=ccrs.PlateCarree());

plt.title('Europe and Middle East Domestic Terrorists Attacks Since 2003', fontsize = 20);

plt.legend(handles=scatter.legend_elements()[0], labels= legendname,fontsize=20, loc='upper right');
x = df_filtered.groupby(['gname'],as_index = False,group_keys = False).filter(lambda x: len(x) >= 50)

x.groupby(['cluster','gname']).count()['eventid']
df_filtered_sa = raw_data[(raw_data['region_txt'] == 'South Asia')& (raw_data['INT_LOG']==0) & (raw_data['iyear']>=2003)]



lat_sa = df_filtered_sa['latitude']

lon_sa = df_filtered_sa['longitude']



data_sa = pd.DataFrame({'latitude':lat_sa, 'longitude': lon_sa}, index = df_filtered_sa.index)

data_sa = data_sa.dropna(how="any")



data_scaled_sa = scaler.fit_transform(data_sa)



db_sa = DBSCAN(eps=0.3, min_samples=100).fit(data_scaled_sa)



labels_sa = db_sa.labels_



data_sa['cluster'] = labels_sa



df_filtered_sa['cluster'] = data_sa['cluster']



# Number of clusters in labels, ignoring noise if present.

n_clusters_sa = len(set(labels_sa)) - (1 if -1 in labels_sa else 0)

n_noise_sa = list(labels_sa).count(-1)



print('Estimated number of clusters: %d' % n_clusters_sa)

print('Estimated number of noise points: %d' % n_noise_sa)
plt.figure(figsize = (20,15))



#extent = [-20, 60, 25, 60]

#central_lon = np.mean(extent[:2])

#central_lat = np.mean(extent[2:])



legendname_sa = []

for items in np.sort(data_sa.cluster.unique()):

    if items == -1:

        legendname_sa.append('Others / Cluster'+str(items))

    else:

        legendname_sa.append('Cluster'+str(items))



#ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))

ax = plt.axes(projection=ccrs.Mercator())

#ax.set_extent(extent,crs=ccrs.PlateCarree())

scatter = ax.scatter(df_filtered_sa.longitude,df_filtered_sa.latitude,transform=ccrs.PlateCarree(), c = df_filtered_sa.cluster,\

                    s = df_filtered_sa.nkill*10);

ax.coastlines(linewidth = 2);

ax.add_feature(cartopy.feature.BORDERS, linestyle="-", linewidth = 2);

ax.add_feature(cartopy.feature.OCEAN);

#ax.add_feature(cartopy.feature.LAND, edgecolor='black')

ax.add_feature(cartopy.feature.LAKES, edgecolor='black');

ax.gridlines(crs=ccrs.PlateCarree());

plt.title('South Asian Domestic Terrorists Attacks Since 2003', fontsize = 20);

plt.legend(handles=scatter.legend_elements()[0], labels= legendname_sa,fontsize=20);
x_sa = df_filtered_sa.groupby('gname',group_keys = False).filter(lambda x: len(x) >= 50)

x_sa.groupby(['cluster','gname']).count()['eventid']

