# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns
import folium
data = pd.read_csv("../input/NYPD_Motor_Vehicle_Collisions.csv")
interesting_columns = ['DATE','TIME','BOROUGH','LATITUDE', 'LONGITUDE',
        'ON STREET NAME', 'CROSS STREET NAME','NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
       'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
       'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
       'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']
data.dtypes
data = data[interesting_columns]
data.head()
danger = data.groupby(["ON STREET NAME", "CROSS STREET NAME"])["NUMBER OF PERSONS INJURED"].sum()
danger.sort_values().tail(6)
# Make a data frame with dots to show on the map
data2 = pd.DataFrame({
'lat':[40.801830,40.663080, 40.658650, 40.716080, 40.640653],
'lon':[-73.931400,-73.962390,-73.890610,-73.803140,  -73.743340],
'name':['East 125th Street, 1 Ave ','Flatbush Ave, Empire Boulevard', 'Pennsylvania Ave, Linden Boulvard', 
        '164th Street, Grand Central Parkway', 'Rockaway Boulevard, Brookville Boulevard'],
'injured people':['94', '95', '96', '115', '191']})
 
# Make an empty map
map1 = folium.Map(location=[40.7348,-73.9060], tiles="OpenStreetMap", zoom_start=11)

# I can add marker one by one on the map
for i in range(0,5):
    folium.Marker([data2.iloc[i]['lat'], data2.iloc[i]['lon']], tooltip=data2.iloc[i]['name']+'<br> Injured people: '+data2.iloc[i]['injured people']).add_to(map1)

folium.TileLayer('Stamen Terrain').add_to(map1)
folium.LayerControl().add_to(map1)


map1
from folium import plugins
from folium.plugins import HeatMap
def get_hour(s):
    if type(s) == str:
        return s[0:2].replace(":", "")
    else:
        return s;
data.head()
data["hour"] = data["TIME"].apply(get_hour) #making new column
data_Oct_2018= data[(data["DATE"].str.startswith("10/")) & (data["DATE"].str.endswith("2018"))]
map2 = folium.Map(location=[40.7508,-73.9060],
                    zoom_start = 11) 
data_Oct_2018 = data_Oct_2018[["hour","LATITUDE","LONGITUDE"]].dropna()
heat_df = data_Oct_2018[['LATITUDE', 'LONGITUDE']]

heat_df["Weight"] = data_Oct_2018["hour"].dropna()
heat_df['Weight'] = heat_df['Weight'].astype(float)
heat_df = heat_df.dropna(axis=0, subset=['LATITUDE','LONGITUDE', "Weight"])

heat_data = [[[row['LATITUDE'],row['LONGITUDE']] for index, row in heat_df[heat_df['Weight'] == i].iterrows()] for i in range(0,24)]
hm = plugins.HeatMapWithTime(heat_data,radius = 13,auto_play=False,max_opacity=0.8)
hm.add_to(map2)
map2













