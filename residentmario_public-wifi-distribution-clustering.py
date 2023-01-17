import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = pd.read_csv("../input/NYC_Wi-Fi_Hotspot_Locations.csv")

data.head()
data['NTAName'].value_counts()
data['Borough'].value_counts().plot.bar()
import folium

display = folium.Map(location=[40.75, -74])



for (_, (lat, long)) in data[['Latitude', 'Longitude']].iterrows():

    folium.CircleMarker([lat, long],

                    radius=5,

                    color='#3186cc',

                    fill_color='#3186cc',

                   ).add_to(display)



display
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, random_state=0).fit(data[['Latitude', 'Longitude']].values)

labels = kmeans.labels_



colors = ['#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd']

display = folium.Map(location=[40.75, -74])



for (lat, long, label) in zip(data['Latitude'], data['Longitude'], labels):

    folium.CircleMarker([lat, long],

                    radius=5,

                    color=colors[label],

                    fill_color=colors[label],

                   ).add_to(display)

    

display
data['Provider'].value_counts()
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