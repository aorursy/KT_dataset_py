# install required modules

!pip install geopy

!pip install folium
# import libraries

import numpy as np 

import pandas as pd 

import folium as fl

import matplotlib.pyplot as plt

import seaborn as sns

from geopy.distance import GreatCircleDistance
#create DataFrame

df = pd.read_csv("../input/bournemouth-venues/bournemouth_venues.csv")

df.head(5)
# rename columns

df = df.rename(columns = {'Venue Latitude':'latitude','Venue Longitude': 'longitude', 'Venue Category': 'category','Venue Name':'place'})



# Visualize the categories

print(df.category.value_counts().iloc[0:11])

print("total categories :",df.category.value_counts().shape)

fig = plt.figure(figsize = (20,5))

sns.countplot(df["category"][0:10])
# Folium map

map = fl.Map([50.720913,-1.879085],zoom_start = 15)



# grouping dataframe by category

df = df.groupby("category")

map
# extracting  all the rows with hotels

hotels = df.get_group('Hotel')

print(hotels.head(5))



# Separating the hotel locations and converting each attribute into list

lat = list(hotels["latitude"])

lon = list(hotels["longitude"])

place = list(hotels["place"])

cat = list(hotels["category"])



# visualize / locate hotels--->Markers in red

for lt,ln,pl,cat in zip(lat,lon,place,cat):

    fl.Marker(location = [lt,ln], tooltip = str(pl) +","+str(cat), icon = fl.Icon(color = 'red')).add_to(map)

map 
parks = df.get_group('Park')



# Separating the all park locations and converting each attribute into list

lat = list(parks["latitude"])

lon = list(parks["longitude"])

place = list(parks["place"])

cat = list(parks["category"])

print(parks)



# parks in green colors

for cat,lt,ln,pl in zip(cat,lat,lon,place):

    fl.Marker(location = [lt,ln], tooltip = str(pl) +","+str(cat), icon = fl.Icon(color = 'green')).add_to(map)

map
beach = df.get_group('Beach')

lat_beach = list(beach["latitude"])

lon_beach = list(beach["longitude"])

place = list(beach["place"])

cat = list(beach["category"])

print(beach)



# Beach in blue color

for cat,lt,ln,pl in zip(cat,lat_beach,lon_beach,place):

    fl.Marker(location = [lt,ln], tooltip = str(pl) +","+str(cat), icon = fl.Icon(color = 'blue')).add_to(map)

map
# latitude and longitude of Hallmark Hotel

Source = (50.718742,-1.890372)



# Empty list to store the distance

distance = []

for lt,ln in zip(lat_beach,lon_beach):

    dist = GreatCircleDistance(Source,(lt,ln))

    distance.append(dist)



# Draw lines between points

for dist,lt,ln in zip(distance,lat_beach,lon_beach):

    if (dist > 0) and (dist <= 0.6):

        fl.PolyLine([Source,(lt,ln),],color = "green", weight = 4).add_to(map)  

    elif (dist > 0.6) and (dist <= 0.9):

        fl.PolyLine([Source,(lt,ln)],color = "orange", weight = 3).add_to(map)

    else :

        fl.PolyLine([Source,(lt,ln)],color = "red", weight = 2).add_to(map)

map
distance