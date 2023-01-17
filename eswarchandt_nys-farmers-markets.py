# essential libraries
import random
from datetime import timedelta  

# storing and anaysis
import numpy as np
import pandas as pd
# Import label encoder 
from sklearn import preprocessing 
# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import folium
from folium import Marker,GeoJson,Choropleth, Circle
from folium.plugins import HeatMap
from folium.plugins import HeatMap, MarkerCluster
# color pallette
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 

# converter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()   

# hide warnings
import warnings
warnings.filterwarnings('ignore')

markets=pd.read_csv(r"../input/nys-farmers-markets-in-new-york-state/farmers-markets-in-new-york-state.csv")
markets.sample(6)
fig = plt.figure(figsize=(20,20))
states = markets.groupby('County')['City'].count().sort_values(ascending=True)
states.plot(kind="barh", fontsize = 20,color='green')
plt.grid(b=True, which='both', color='black')
plt.xlabel('No of cities taken from each state', fontsize = 20)
plt.show ()
markets.skew()
markets.kurtosis()
markets=markets.dropna()

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in columns
markets['FMNP']= label_encoder.fit_transform(markets['FMNP']) 
markets['SNAP']= label_encoder.fit_transform(markets['SNAP']) 
markets['FCC Issued']= label_encoder.fit_transform(markets['FCC Issued']) 
markets['FCC Accepted']= label_encoder.fit_transform(markets['FCC Accepted']) 
markets.sample(4)
# World wide
m_1 = folium.Map(location=['42.9223','-78.8969'], tiles='cartodbpositron', zoom_start=6)

# Add points to the map
for idx, row in markets.iterrows():
    Marker([row['Latitude'], row['Longitude']]).add_to(m_1)

# Display the map
m_1
# Create map with release incidents and monitoring stations
m = folium.Map(location=[48,-71], zoom_start=5)
HeatMap(data=markets[['Latitude', 'Longitude']], radius=15).add_to(m)

# Show the map
m
import math
m = folium.Map(location=[48,-71], tiles='cartodbpositron', zoom_start=5)

# Add points to the map
mc = MarkerCluster()
for idx, row in markets.iterrows():
    if not math.isnan(row['Longitude']) and not math.isnan(row['Latitude']):
        mc.add_child(Marker([row['Latitude'], row['Longitude']]))
m.add_child(mc)

# Display the map
m