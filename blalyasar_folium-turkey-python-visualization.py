# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import folium
map1 = folium.Map(location=[36.8, 30.7],
                    zoom_start = 10)
map1
df = pd.read_csv("/kaggle/input/store-locations/directory.csv")
df.head()
df.tail()
df = df[(df['Country'] == "TR")]
df.head()
df.describe()
# Plotting a bar graph of the number of stores in each city, for the first ten cities listed
# in the column 'City'
city_count  = df['City'].value_counts()
city_count = city_count[:10,]
plt.figure(figsize=(10,5))
sns.barplot(city_count.index, city_count.values, alpha=0.8)
plt.title('Starbucks İlk 10 Sehir')
plt.ylabel('Mağaza sayisi', fontsize=12)
plt.xlabel('Sehir', fontsize=12)
plt.show()
import folium
#Create a map
m = folium.Map(location=[39.7,34.7], control_scale=True, zoom_start=6,attr = "text some")
df_copy = df.copy()
# loop through data to create Marker for each hospital
for i in range(0,len(df_copy)):
    
    # html to be displayed in the popup 
    html="""
    <h4> ADRES: </h4>""" + str( df_copy.iloc[i]['Street Address'])
    
    #IFrame 
    iframe = folium.IFrame(html=html, width=150, height=250)
    popup = folium.Popup(iframe)
    

    folium.Marker(
    location=[df_copy.iloc[i]['Latitude'], df_copy.iloc[i]['Longitude']],
    popup=popup,
    tooltip=str(df_copy.iloc[i]['City']),
    icon=folium.Icon(color='lightblue',icon='medkit',prefix="fa"),
    ).add_to(m)
        
#m.save("hospitals-location-tr.html")
m
