import numpy as np 

import pandas as pd 



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import squarify



# for providing path

import os


crimeData1 = pd.read_csv('../input/sanfranciso-crime-dataset/Police_Department_Incidents_-_Previous_Year__2016_.csv')

print (crimeData1.shape) #shape of my dataset

crimeData1.head() #printing first five rows
crimeData1.info()
# To find stats of each feature.

crimeData1.describe()
crimeData1.isnull().sum()
# different categories of crime



plt.rcParams['figure.figsize'] = (20, 9)

plt.style.use('dark_background')



sns.countplot(crimeData1['Category'], palette = 'gnuplot')



plt.title('Major Crimes in Sanfrancisco', fontweight = 30, fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
# plotting a tree map



y = crimeData1['Category'].value_counts().head(25)

    

plt.rcParams['figure.figsize'] = (15, 15)

plt.style.use('fivethirtyeight')



color = plt.cm.magma(np.linspace(0, 1, 15))

squarify.plot(sizes = y.values, label = y.index, alpha=.8, color = color)

plt.title('Tree Map for Top 25 Crimes', fontsize = 20)



plt.axis('off')

plt.show()
# Regions with count of crimes



plt.rcParams['figure.figsize'] = (20, 9)

plt.style.use('seaborn')



color = plt.cm.spring(np.linspace(0, 1, 15))

crimeData1['PdDistrict'].value_counts().plot.bar(color = 'r', figsize = (15, 10))



plt.title('Places with Most Crime',fontsize = 30,color='g')



plt.xticks(rotation = 90)

plt.show()
# Regions with count of crimes



plt.style.use('seaborn')





crimeData1['DayOfWeek'].value_counts().head(15).plot.pie(figsize = (15, 8), explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))



plt.title('Crime count on each day',fontsize = 20)



plt.xticks(rotation = 90)

plt.show()
# checking the time at which crime occurs mostly



import warnings

warnings.filterwarnings('ignore')



color = plt.cm.twilight(np.linspace(0, 5, 100))

crimeData1['Time'].value_counts().head(20).plot.bar(color = color, figsize = (15, 9))



plt.title('Distribution of crime over the day', fontsize = 20)

plt.show()


df = pd.crosstab(crimeData1['Category'], crimeData1['PdDistrict'])

color = plt.cm.Greys(np.linspace(0, 1, 10))



df.div(df.sum(1).astype(float), axis = 0).plot.bar(stacked = True, color = color, figsize = (18, 12))

plt.title('District vs Category of Crime', fontweight = 30, fontsize = 20)



plt.xticks(rotation = 90)

plt.show()
t = crimeData1.PdDistrict.value_counts()



table = pd.DataFrame(data=t.values, index=t.index, columns=['Count'])

table = table.reindex(["CENTRAL", "NORTHERN", "PARK", "SOUTHERN", "MISSION", "TENDERLOIN", "RICHMOND", "TARAVAL", "INGLESIDE", "BAYVIEW"])



table = table.reset_index()

table.rename({'index': 'Neighborhood'}, axis='columns', inplace=True)



table
gjson = r'https://cocl.us/sanfran_geojson'

sf_map = folium.Map(location = [37.77, -122.42], zoom_start = 12)

sf_map.choropleth(

    geo_data=gjson,

    data=table,

    columns=['Neighborhood', 'Count'],

    key_on='feature.properties.DISTRICT',

    fill_color='YlOrRd', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Crime Rate in San Francisco'

)



sf_map

#it will run on jupyter notebook with internet connection
# Let's limit the dataset to the first 15,000 records for this example

data =crimeData1



# Store our latitude and longitude

latitudes = data["Y"]

longitudes = data["X"]



# Creating the location we would like to initialize the focus on. 

# Parameters: Lattitude, Longitude, Zoom

gmap = gmplot.GoogleMapPlotter(37.77, -122.365565, 12)



# Overlay our datapoints onto the map

gmap.heatmap(latitudes, longitudes)



# Generate the heatmap into an HTML file

gmap.draw("my_heatmap.html")

#it will also going to run on jupyter notebook
from folium.plugins import HeatMap



df = pd.read_csv('crime1.csv')



# X und Y not null

df = df[(df['X'] != 0) & (df['Y'] != 0)]



# Create dataset, crime = burglary

crime_vehicle_theft =  df[(df['Category'] == "VEHICLE THEFT")]

#testvehth = vehth[:50]



# Create map, centre of San Francisco

map_van = folium.Map(location = [37.773972, -122.431297], zoom_start = 10)



# Create list with x- and y-coordinates, add to map, show map

heat_data = [[row['Y'],row['X']] for index, row in crime_vehicle_theft.iterrows()]

HeatMap(heat_data).add_to(map_van)



map_van