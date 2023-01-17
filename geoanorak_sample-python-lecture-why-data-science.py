# THIS IS A COMMENT IN PYTHON - it is for people to read and Python ignores it
# Getting Started with Python
# Why Python? - Why data Science?

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

df = pd.read_csv("http://uoweb3.ncl.ac.uk/api/v1.1/sensors/PER_AIRMON_MONITOR1135100/data/csv/?data_variable=NO2&last_n_days=1")
display(df.head())
df.columns

df['Value'].count()
df.Timestamp = pd.to_datetime(df.Timestamp)
df = df.set_index(df.Timestamp)

plt.plot(df.index, df.Value)
plt.hlines(40, df.index.min(), df.index.max())
plt.show()
df.describe()

lat = df['Sensor Centroid Latitude']
long = df['Sensor Centroid Longitude']
lat = lat[0]
long = long[0]
print(long, lat)

import folium
from folium import Marker

# Create a map
m_1 = folium.Map(location=[lat, long], tiles='openstreetmap', zoom_start=10)

Marker((lat,long)).add_to(m_1)

# Display the map
m_1

df=pd.read_csv('http://uoweb3.ncl.ac.uk/api/v1.1/sensors/data/csv/?last_n_days=1&broker=aq_mesh_api&sensor_type=NO2&data_variable=NO2')
display(df.head())
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.set_index(df['Timestamp'])

len(df['Sensor Name'].unique())
df.info()
res_mean = df.groupby(df['Sensor Name']).resample('48H').mean()
display(res_mean) 

m_2 = folium.Map(location=[lat, long], tiles='cartodbpositron', zoom_start=13)

# Add points to the map
for idx, row in res_mean.iterrows():
    Marker([row['Sensor Centroid Latitude'], row['Sensor Centroid Longitude']]).add_to(m_2)


# Display the map
m_2

from folium.plugins import HeatMap
m_5 = folium.Map(location=[lat,long], tiles='stamenwatercolor', zoom_start=12)

# Add a heatmap to the base map
HeatMap(data=res_mean[['Sensor Centroid Latitude', 'Sensor Centroid Longitude', 'Value']], radius=35).add_to(m_5)

# Display the map
m_5