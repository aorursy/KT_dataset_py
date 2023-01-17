# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_json("/kaggle/input/cities-of-turkey/cities_of_turkey.json")
df.head()
region = df["region"]
region.sample()
import folium
#Create a map
m = folium.Map(location=[40,35], control_scale=True, zoom_start=7,attr = "text some")
df_copy = df.copy()

# loop through data to create Marker for each hospital
for i in range(0,len(df_copy)):
    html="""
    <h4> name: </h4>""" + str(df_copy.iloc[i]['name'])
    
    iframe = folium.IFrame(html=html, width=200, height=300)
    

    folium.Marker(
    location=[df_copy.iloc[i]['latitude'], df_copy.iloc[i]['longitude']],
    
    tooltip=str(df_copy.iloc[i]['name']),
    icon=folium.Icon(color='orange',icon='medkit',prefix="fa"),
    ).add_to(m)
        

m
def Renklendir(bolgerenk):
    if bolgerenk == "Akdeniz":
        return "beige"
    elif bolgerenk == "Ege":
        return "blue"
    elif bolgerenk == "Doğu Anadolu":
        return "red"
    elif bolgerenk == "Karadeniz":
        return "orange"
    elif bolgerenk == "Güneydoğu Anadolu":
        return "yellow"
    elif bolgerenk == "Marmara":
        return "black"
    elif bolgerenk == "İç Anadolu":
        return "darkpurple"

df_copy


# loop through data to create Marker for each hospital
for i in range(0,len(df_copy)):
    html="""
    <h4> name: </h4>""" + str(df_copy.iloc[i]['name'])

    iframe = folium.IFrame(html=html, width=200, height=300)
    renk = Renklendir(str(df_copy.iloc[i]['region']))
    
    folium.Marker(
    location=[df_copy.iloc[i]['latitude'], df_copy.iloc[i]['longitude']],
    
    tooltip=str(df_copy.iloc[i]['name']),
    icon=folium.Icon(color=renk,icon='cloud',prefix="fa"),
        #icon=folium.Icon(icon='cloud')info-sign
    ).add_to(m)
m
"""
for bolgerenk in region:
    renk = Renklendir(str(bolgerenk))
    nufusHaritasi.add_child(folium.Marker(location=[en,boy],tooltip=(il+"  nüfus:"+nuf),icon=folium.Icon(color=renk)))
"""
