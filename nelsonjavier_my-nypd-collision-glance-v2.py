# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt #visualize into notebook
%matplotlib inline  
pd.__version__
print(os.listdir("../input"))
COLLISIONS_CSV = '../input/nypd-motor-vehicle-collisions.csv'
DATADICTIONARY = '../input/Collision_DataDictionary.xlsx'
METADATA_JSON = '../input/socrata_metadata.json'
collisionsDF = pd.read_csv(COLLISIONS_CSV)
collisionsDF.tail(2)
collisionsDF.head(2)
collisionsDF.info()
# Assuming that dataframe has DATE and Time columns sorted (meaning row 0 is latest date and last row represents the earliest date)
# It can be suppossed based on the tail() and head() results
oldest_date = collisionsDF.iloc[-1].DATE #tail
newest_date = collisionsDF.iloc[0].DATE #head
oldest_date
newest_date
# Filter the dataframes into number of injured and number of deaths which are motorist and where there is a defined location
injured_mot = collisionsDF[collisionsDF['NUMBER OF MOTORIST INJURED'] > 0 & collisionsDF['LATITUDE'].notnull()]
killed_mot = collisionsDF[collisionsDF['NUMBER OF MOTORIST KILLED'] > 0 & collisionsDF['LATITUDE'].notnull()]
# Now plot these dataframes per borough
fig = plt.figure(figsize=(12,8))

bx = fig.add_subplot(111)
bx2 = bx.twinx() # Create another axes that shares the same x-axis as bx.
injured_mot.BOROUGH.value_counts().plot(kind='bar', color='blue', width=0.4, position=0, ax=bx)
killed_mot.BOROUGH.value_counts().plot(kind='bar', color='red', width=0.4, position=1, ax=bx2)
bx.set_title('Number of motorist injured and killed per borough', fontsize=20, fontweight='bold')
bx.set_ylabel('Number of motorist injured', color='blue')
bx2.set_ylabel('Number of motorist killed', color='red')

plt.show()
# plot the collisions data on a map for visualization

import folium
from folium import plugins
from io import StringIO
import folium 
collisionsDF['LATITUDE'][0]
collisionsDF['LONGITUDE'][0]
m = folium.Map(location=[collisionsDF['LATITUDE'][0], collisionsDF['LONGITUDE'][0]], zoom_start=11,tiles='stamentoner')
m # Calls the map to display
collisions = injured_mot['NUMBER OF MOTORIST INJURED'].head(1000)
map_data = collisions

interval = (max(map_data)-min(map_data))/255.0
red_val = ((map_data-min(map_data))/interval).astype('int')

interval
max(red_val)
min(red_val)
injured_mot['LATITUDE'][ injured_mot['LATITUDE'].notnull()].head()
injured_mot['LONGITUDE'][ injured_mot['LONGITUDE'].notnull()].head()
#show points into the map
for lat, long, red in zip(injured_mot['LATITUDE'][ injured_mot['LATITUDE'].notnull()], injured_mot['LONGITUDE'][ injured_mot['LONGITUDE'].notnull()], red_val):
    folium.CircleMarker([lat, long], color='#0000ff', fill=True, radius=2).add_to(m)
    #colourString = '#%0.2x00%0.2x' % (red, (255-red))
    #folium.CircleMarker([lat, long], color=colourString, fill=True, radius=2).add_to(m)
m
collisions = killed_mot['NUMBER OF MOTORIST INJURED'].head(1000)
map_data = collisions

interval = (max(map_data)-min(map_data))/255.0
red_val = ((map_data-min(map_data))/interval).astype('int')

interval
#show points into the map
for lat, long, red in zip(killed_mot['LATITUDE'][ killed_mot['LATITUDE'].notnull()], killed_mot['LONGITUDE'][ killed_mot['LONGITUDE'].notnull()], red_val):
    folium.CircleMarker([lat, long], color='#f8340a', fill=True, radius=2).add_to(m)
    #colourString = '#%0.2x00%0.2x' % (red, (255-red))
    #folium.CircleMarker([lat, long], color=colourString, fill=True, radius=2).add_to(m)
m