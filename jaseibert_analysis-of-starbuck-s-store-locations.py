import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import folium 
from folium import plugins
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input"))
#Read in the Store Directory file 
SB_Df = pd.read_csv('../input/directory.csv')

#Transform the CSV file into a DataFrame
SB_Df= pd.DataFrame(SB_Df)
SB_Df.head(5)
SB_Df.notnull().sum() * 100/SB_Df.shape[0]
SB_Df.Country.value_counts().head(20) 
SB_Df.City.value_counts().head(20)
SB_Df["Ownership Type"].value_counts().head(4)
figure = plt.figure(figsize=(22,6))
axes = figure.add_subplot(111)
axes.set(title = "The Top 20 Countries with the most Starbucks")
SB_Df.Country.value_counts().head(20).plot(kind="bar", color = "green")
figure = plt.figure(figsize=(17,8))
axes = figure.add_subplot(111)
axes.set(title = "The Top 20 Cities in the World with the most Starbucks")
SB_Df.City.value_counts().head(20).plot(kind="bar", color = "green")
figure = plt.figure(figsize=(10,10))
axes = figure.add_subplot(111)
axes.set(title = "What type of Store is it?")
SB_Df['Ownership Type'].value_counts().plot(kind="pie", colors = ["green", "blue", "red", "yellow"])
plt.show()
#Find all the NaN Values in the DataFrame
SB_Df.isnull().sum()
#Combine the Lat & Long values into a location var
location = SB_Df[['Latitude','Longitude']]

#Transform the NaN values into (0,0) 
location.fillna(0, inplace=True)

#Transform the location values into a list
locationList = location.values.tolist()
len(locationList)

#check random var
locationList[223]
#Define initial map parameters
Evansville_Coordinates = (37.987734, -87.534703)

#Create empty map zoomed in on Evansville
map = folium.Map(location= Evansville_Coordinates, tiles='CartoDB dark_matter', zoom_start=11)
marker_cluster = folium.plugins.MarkerCluster().add_to(map)

#Create a for loop & itterate through each store location 
for loc in range(0, len(locationList)):
    folium.Marker(locationList[loc], popup=SB_Df['Store Number'][loc], icon=folium.Icon(color='green', icon_color='white', icon='info-sign')).add_to(marker_cluster)
map