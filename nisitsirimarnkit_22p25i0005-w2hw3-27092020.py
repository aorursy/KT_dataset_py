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
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df
df.info()
df['neighbourhood'].value_counts()
df.isnull().sum()
df_new = df[['neighbourhood','price','latitude','longitude']] 
df_new = df_new.groupby('neighbourhood').agg({'price':'mean','latitude':'mean','longitude': 'mean'})
data1 = df_new
data1
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler


fig = plt.figure(figsize=(30,27))
dendogram = sch.dendrogram(sch.linkage(data1,method='ward'),leaf_rotation=90, leaf_font_size=12,labels=data1.index) 
plt.title("Dendrograms")  
plt.show()
plt.figure(figsize=(30, 27))  
plt.title("Dendrograms")  
dend = sch.dendrogram(sch.linkage(data1, method='ward'),leaf_rotation=90, leaf_font_size=12,labels=data1.index)
plt.axhline(y=1000, color='r', linestyle='--')
plt.show()
hc = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(data1)
print(y_hc)
    
fig = plt.figure(figsize=(20,18))
plt.scatter(data1.index,data1['price'], c=y_hc) 
plt.title('K = 2')
plt.xlabel('neighbourhood')
plt.xticks(rotation=90)
plt.ylabel('price')
plt.show()
import folium
from folium.plugins import MarkerCluster
cluster_map = folium.Map(location=[40.64749,-73.97237],tiles='cartodbpositron',zoom_start=10) 

for i in range(len(y_hc)):
    lat = df_new.iloc[i]['latitude']
    long = df_new.iloc[i]['longitude']
    radius = 6

    if y_hc[i] == 0:
        folium.CircleMarker(location = [lat, long], radius=radius,  fill =True, color='red').add_to(cluster_map)
    elif y_hc[i] == 1:
        folium.CircleMarker(location = [lat, long], radius=radius,  fill =True, color='blue').add_to(cluster_map)
    else:
        pass

cluster_map

h_map = folium.Map(location=[40.64749,-73.97237],zoom_start = 10) 

df_new['latitude'] = df_new['latitude'].astype(float)
df_new['longitude'] = df_new['longitude'].astype(float)

heat_df = df_new[['latitude', 'longitude']]
heat_df = heat_df.dropna(axis=0, subset=['latitude','longitude'])

heat_data = [[row['latitude'],row['longitude']] for index, row in heat_df.iterrows()]

HeatMap(heat_data).add_to(h_map)

h_map
