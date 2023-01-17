!pip install git+https://github.com/python-visualization/branca

!pip install git+https://github.com/sknzl/folium@update-css-url-to-https
#dependencies

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting

import seaborn as sns #for beatiful visualization

import folium

from folium import plugins



#set file path

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

folium.__version__
folium.branca.__version__
fire_nrt_m6 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_nrt_M6_101673.csv")

fire_archive_m6 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_archive_M6_101673.csv")

fire_nrt_v1 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_nrt_V1_101674.csv")

fire_archive_v1 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_archive_V1_101674.csv")



type(fire_nrt_v1)
df_merged = pd.concat([fire_archive_v1,fire_nrt_v1],sort=True)

data = df_merged

data.head()
data.info()
df_filter = data.filter(["latitude","longitude","acq_date","frp"])

df_filter.head()
df = df_filter[df_filter['acq_date']>='2019-11-01']

df.head()
data_topaffected = df.sort_values(by='frp',ascending=False).head(10)

data_topaffected
#Create a map

m = folium.Map(location=[-35.0,144], control_scale=True, zoom_start=3,attr = "text some")

df_copy = data_topaffected.copy()



# loop through data to create Marker for each hospital

for i in range(0,len(df_copy)):

    

    folium.Marker(

    location=[df_copy.iloc[i]['latitude'], df_copy.iloc[i]['longitude']],

    #popup=popup,

    tooltip="frp: " + str(df_copy.iloc[i]['frp']) + "<br/> date: "+ str(df_copy.iloc[i]['acq_date']),

    icon=folium.Icon(color='red',icon='fire',prefix="fa"),

    ).add_to(m)

        

m

dfdate = df[['acq_date','frp']].set_index('acq_date')

dfdate_highest = dfdate.groupby('acq_date').sum().sort_values(by='frp',ascending=False)

dfdate_highest.head(10)
plt.figure(figsize=(10,5))

sns.set_palette("pastel")

ax = sns.barplot(x='acq_date',y='frp',data=df)

for ind, label in enumerate(ax.get_xticklabels()):

    if ind % 10 == 0:  # every 10th label is kept

        label.set_visible(True)

    else:

        label.set_visible(False)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.xlabel("Date")

plt.ylabel('FRP (fire radiation power)')

plt.title("time line of bushfire in Australia")

plt.tight_layout()

from folium.plugins import HeatMapWithTime

# A small function to get heat map with time given the data



def getmap(ip_data,location,zoom,radius):

    

    #get day list

    dfmap = ip_data[['acq_date','latitude','longitude','frp']]

    df_day_list = []

    for day in dfmap.acq_date.sort_values().unique():

        df_day_list.append(dfmap.loc[dfmap.acq_date == day, ['acq_date','latitude', 'longitude', 'frp']].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist())

    

    # Create a map using folium

    m = folium.Map(location, zoom_start=zoom,tiles='Stamen Terrain')

    #creating heatmap with time

    HeatMapWithTime(df_day_list,index =list(dfmap.acq_date.sort_values().unique()), auto_play=False,radius=radius, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}, min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(m)



    return m

getmap(df,[-27,132],3.5,3)

#df tail for the latest data

df_10days = df.tail(21500)

#Using getmap function to obtain map from above, location set to canberra

getmap(df_10days,[-35.6,149.12],8,3)

#Using getmap function to obtain map from above, location set to kangaroo island

getmap(df,[-36, 137.22],8.5,3)
