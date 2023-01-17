import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
%%time

list_df = []

chunksize = 10**6



for chunk in pd.read_csv('../input/chicago-rideshare/rideshare.csv',index_col=0,chunksize=chunksize):

    df = chunk

    list_df.append(df)
rides = pd.concat(list_df)

rides.info()
rides.isnull().sum()
rides.dropna(inplace=True)

rides.reset_index(drop=True,inplace=True)

rides.columns = rides.columns.str.replace(' ', '_').str.lower()
%%time

rides['trip_start_timestamp'] =pd.to_datetime(rides['trip_start_timestamp'],format= '%m/%d/%Y %I:%M:%S %p')
census = pd.read_csv('../input/chicago-census-socioecon-commarea-2012/Chicago_Census_SociaEcon_CommArea_2008_2012.csv')

census.columns = census.columns.str.replace(' ','_').str.lower()

census.columns
comm_dict = pd.Series(census.community_area_name,index=census.community_area_number).to_dict()

rides['pickup_community_area_name'] = rides['pickup_community_area'].map(lambda x: comm_dict[x])

rides['dropoff_community_area_name'] = rides['dropoff_community_area'].map(lambda x: comm_dict[x])
rides['weekday'] = rides['trip_start_timestamp'].map(lambda x: x.weekday())
rides['time'] = 0

rides.loc[rides['trip_start_timestamp'].dt.hour <= 4,'time'] = 5

rides.loc[(rides['trip_start_timestamp'].dt.hour > 4)&((rides['trip_start_timestamp'].dt.hour <= 8)),'time'] = 0

rides.loc[(rides['trip_start_timestamp'].dt.hour > 8)&((rides['trip_start_timestamp'].dt.hour <= 12)),'time'] = 1

rides.loc[(rides['trip_start_timestamp'].dt.hour > 12)&((rides['trip_start_timestamp'].dt.hour <= 16)),'time'] = 2

rides.loc[(rides['trip_start_timestamp'].dt.hour > 16)&((rides['trip_start_timestamp'].dt.hour <= 20)),'time'] = 3

rides.loc[(rides['trip_start_timestamp'].dt.hour > 20),'time'] = 4
rides_by_time = rides.groupby('time')['trip_id'].count().reset_index(name = 'trips')

rides_by_day = rides.groupby('weekday')['trip_id'].count().reset_index(name = 'trips')
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

rides_by_time.plot(kind='bar',y='trips',x='time',legend=None,figsize=(18,6),color=sns.color_palette('Blues_d'),ax=ax1)

ax1.set_xticklabels(labels= ["04:00-08:00","08:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-00:00","00:00-04:00"],rotation=60)

ax1.set_xlabel('Time of Day')

ax1.set_ylabel('Number of Trips')

ax1.set_title('Rides by time of day')

rides_by_day.plot.bar(x='weekday',y='trips',legend=None,figsize=(18,6),ax=ax2,color=sns.color_palette('YlOrRd_d'))

ax2.set_xticklabels(labels= ["Monday", "Tuesday", "Wednesday","Thursday", "Friday","Saturday","Sunday"],rotation=60)

ax2.set_xlabel('Day of the Week')

ax2.set_ylabel('Number of Trips')

ax2.set_title('Rides by day of the week')

fig.subplots_adjust(wspace=0.3)

plt.savefig('rides_day_time.jpg')

fig.savefig('rides_day_time.pdf')
sample = rides.sample(frac=0.1)

sample['hour'] = sample['trip_start_timestamp'].dt.hour

fig, ax= plt.subplots()

sns.boxplot(x='weekday',y='hour',data=sample)

ax.set_xticklabels(labels= ["Monday", "Tuesday", "Wednesday","Thursday", "Friday","Saturday","Sunday"],rotation=60)

ax.set_xlabel('Days')

ax.set_ylabel('Time of Day')

ax.set_title('Trip time boxplot by day')

fig.savefig('trip_time_boxplot.jpg')
import folium

from folium import IFrame, FeatureGroup, LayerControl, Map, Marker, plugins
dropoff_locations = rides.groupby('dropoff_community_area')[['dropoff_community_area_name','dropoff_centroid_latitude','dropoff_centroid_longitude']].first().reset_index()

pickup_locations = rides.groupby('pickup_community_area')[['pickup_community_area_name','pickup_centroid_latitude','pickup_centroid_longitude']].first().reset_index()
pickup_locations['trips'] = rides.groupby('pickup_community_area')['trip_id'].count().values

dropoff_locations['trips'] = rides.groupby('dropoff_community_area')['trip_id'].count().values
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

pickup_locations.sort_values('trips',ascending=False).head(20).plot(kind='bar',y='trips',x='pickup_community_area_name',legend=None,figsize=(15,15),

                                                                    color=sns.cubehelix_palette(20,start=3,rot=-.25,reverse=True),ax=ax1)

#ax1.set_xticklabels(labels= ["04:00-08:00","08:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-00:00","00:00-04:00"],rotation=60)

ax1.set_xlabel('Community Area')

ax1.set_ylabel('Number of Pickups')

ax1.set_title('Most Rides by Community Area (Pickups)')

dropoff_locations.sort_values('trips',ascending=False).head(20).plot(kind='bar',x='dropoff_community_area_name',y='trips',legend=None,figsize=(15,15),ax=ax2,

                                                                     color=sns.cubehelix_palette(20,reverse=True))

#ax2.set_xticklabels(labels= ["Monday", "Tuesday", "Wednesday","Thursday", "Friday","Saturday","Sunday"],rotation=60)

ax2.set_xlabel('Community Area')

ax2.set_ylabel('Number of Trips')

ax2.set_title('Most Rides by Community Area (Dropoffs)')

fig.subplots_adjust(hspace=0.5)

plt.savefig('rides_by_area.jpg')

fig.savefig('rides_by_area.pdf')
pickup_locations.head()
Chicago_COORDINATES = (41.895140898, -87.624255632)

chicago_pickup_map = folium.Map(location=Chicago_COORDINATES,

                        zoom_start=11,tiles='CartoDB positron')





for i in range(len(pickup_locations)):

    lat = pickup_locations.iloc[i][2]

    long = pickup_locations.iloc[i][3]

    radius = 1.5*np.log(pickup_locations['trips'].iloc[i])

    if pickup_locations['trips'].iloc[i] >= 10**6:

        color = 'navy'

    elif pickup_locations['trips'].iloc[i] >=10**5:

        color = 'royalblue'

    else:

        color = 'lightseagreen'

    popup_text = """Community Area : {}<br>

                Pickups : {}<br>"""

    popup_text = popup_text.format(pickup_locations.iloc[i][1],

                              pickup_locations['trips'].iloc[i])

    folium.CircleMarker(location = [lat, long], radius=radius,popup= popup_text,color=color, fill = True).add_to(chicago_pickup_map)
chicago_pickup_map.save('chicago_pickups_html')
chicago_dropoff_map = folium.Map(location=Chicago_COORDINATES,

                        zoom_start=11,tiles='CartoDB positron')





for i in range(len(dropoff_locations)):

    lat = dropoff_locations.iloc[i][2]

    long = dropoff_locations.iloc[i][3]

    radius = 1.5*np.log(dropoff_locations['trips'].iloc[i])

    if dropoff_locations['trips'].iloc[i] >= 10**6:

        color = 'darkred'

    elif dropoff_locations['trips'].iloc[i] >= 10**5:

        color='red'

    else:

        color = 'salmon'

    popup_text = """Community Area : {}<br>

                Dropoffs : {}<br>"""

    popup_text = popup_text.format(dropoff_locations.iloc[i][1],

                              dropoff_locations['trips'].iloc[i])

    folium.CircleMarker(location = [lat, long], radius=radius,popup= popup_text,color=color, fill = True).add_to(chicago_dropoff_map)
chicago_dropoff_map.save('chicago_dropoff.html')
chmap = folium.Map(location=Chicago_COORDINATES, zoom_start=11,tiles='CartoDB positron')

for i in range(len(pickup_locations)):

    lat = pickup_locations.iloc[i][2]

    long = pickup_locations.iloc[i][3]

    radius = 1.5*np.log(pickup_locations['trips'].iloc[i])

    if pickup_locations['trips'].iloc[i] >= 10**6:

        color = 'navy'

    elif pickup_locations['trips'].iloc[i] >=10**5:

        color = 'royalblue'

    else:

        color = 'lightseagreen'

    popup_text = """Community Area : {}<br>

                Pickups : {}<br>"""

    popup_text = popup_text.format(pickup_locations.iloc[i][1],

                              pickup_locations['trips'].iloc[i])

    folium.CircleMarker(location = [lat, long], radius=radius,popup= popup_text,color=color, fill = True).add_to(chmap)



for i in range(len(dropoff_locations)):

    lat = dropoff_locations.iloc[i][2]

    long = dropoff_locations.iloc[i][3]

    radius = 1.5*np.log(dropoff_locations['trips'].iloc[i])

    if dropoff_locations['trips'].iloc[i] >= 10**6:

        color = 'darkred'

    elif dropoff_locations['trips'].iloc[i] >= 10**5:

        color='red'

    else:

        color = 'salmon'

    popup_text = """Community Area : {}<br>

                Dropoffs : {}<br>"""

    popup_text = popup_text.format(dropoff_locations.iloc[i][1],

                              dropoff_locations['trips'].iloc[i])

    folium.CircleMarker(location = [lat, long], radius=radius,popup= popup_text,color=color, fill = True).add_to(chmap)
chmap
chmap.save('chicago_rides.html')
taxi = pd.read_csv('../input/chicago-taxi-2017/taxi2017.csv',index_col=0)

taxi.head()