import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import folium
from sklearn.cluster import KMeans
#data = pd.read_csv("../input/nyc-public-wifi/NYC_Wi-Fi_Hotspot_Locations.csv")
data = pd.read_csv("../input/nyc-wi-fi-hotspot-locations.csv")
data.head()
from sklearn.cluster import KMeans
selection = data[data['Provider'] == 'LinkNYC - Citybridge']
kmeans = KMeans(n_clusters=5, random_state=0).fit(selection[['Latitude', 'Longitude']].values)
labels = kmeans.labels_

colors = ['#d7191c','#fdae61','#ffffbf','#abdda4','#2b83ba']
display = folium.Map(location=[40.75, -74])


for (lat, long, label) in zip(selection['Latitude'], selection['Longitude'], labels):
    folium.CircleMarker([lat, long],
                    radius=5,
                    color=colors[label],
                    fill_color=colors[label],
                   ).add_to(display)
    
display
data['Borough'].value_counts().plot.bar()
data['Provider'].value_counts()