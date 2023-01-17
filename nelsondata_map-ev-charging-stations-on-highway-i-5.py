# import Pandas to process the data and Folium for drawing maps

import pandas as pd
import folium
from folium.plugins import MarkerCluster

pd.set_option('display.max_columns',23)

d = pd.read_csv("/kaggle/input/doe-ev-charging-stations-i5/stations.csv")

# select data to explore
df = d[["Station Name","Street Address","City","State","ZIP",
        "Geocode Status","Latitude","Longitude",
        "Date Last Confirmed","Groups With Access Code","Access Days Time",
        "Cards Accepted","Access Code","Access Detail Code",
        "EV Level1 EVSE Num","EV Level2 EVSE Num","EV DC Fast Count",
        "EV Network","EV Pricing","EV Connector Types"]]
# get a list of the EV charging stations in California along with location and GPS coordinates

Location = df["Station Name"].loc[df['State'] =='CA'].tolist()
Latitude = df["Latitude"].loc[df['State'] =='CA'].tolist()
Longitude = df["Longitude"].loc[df['State'] =='CA'].tolist()
print("Number of Stations in California ", len(Location))

# create a Pandas dataframe and load these lists

sm = pd.DataFrame(columns = {'Location','Latitude','Longitude'})
sm.columns = ['Location','Latitude','Longitude']
sm['Location']  = Location
sm['Latitude']  = Latitude
sm['Longitude'] = Longitude

# use the dataframe to draw the map
# center the map on the mean of latitude and longitude 
# use the OpenStreetMap map format and set map view using zoom_start
map_world = folium.Map(location=[sm.Latitude.mean(), sm.Longitude.mean()], tiles = 'OpenStreetMap', zoom_start = 6)

#  add Locations to map
for lat, lng, label in zip(sm.Latitude, sm.Longitude, sm.Location):
    folium.CircleMarker(
        [lat, lng],
        radius=1,
        popup=label,
        fill=True,
        color='Blue',
        fill_color='Yellow',
        fill_opacity=0.6
        ).add_to(map_world)

map_world
# gather the location of stations in Oregon

Location = df["Station Name"].loc[df['State'] =='OR'].tolist()
Latitude = df["Latitude"].loc[df['State'] =='OR'].tolist()
Longitude = df["Longitude"].loc[df['State'] =='OR'].tolist()
print("Number of Stations in Oregon ", len(Location))

sm = pd.DataFrame(columns = {'Location','Latitude','Longitude'})
sm.columns = ['Location','Latitude','Longitude']
sm['Location']  = Location
sm['Latitude']  = Latitude
sm['Longitude'] = Longitude

map_world = folium.Map(location=[sm.Latitude.mean(), sm.Longitude.mean()], tiles = 'OpenStreetMap', zoom_start = 6)

#  map the locations in Oregon
for lat, lng, label in zip(sm.Latitude, sm.Longitude, sm.Location):
    folium.CircleMarker(
        [lat, lng],
        radius=1,
        popup=label,
        fill=True,
        color='Blue',
        fill_color='Yellow',
        fill_opacity=0.6
        ).add_to(map_world)

map_world
# gather the location of stations in Washington

Location = df["Station Name"].loc[df['State'] =='WA'].tolist()
Latitude = df["Latitude"].loc[df['State'] =='WA'].tolist()
Longitude = df["Longitude"].loc[df['State'] =='WA'].tolist()
print("Number of Stations in Washington ", len(Location))

sm = pd.DataFrame(columns = {'Location','Latitude','Longitude'})
sm.columns = ['Location','Latitude','Longitude']
sm['Location']  = Location
sm['Latitude']  = Latitude
sm['Longitude'] = Longitude

map_world = folium.Map(location=[sm.Latitude.mean(), sm.Longitude.mean()], tiles = 'OpenStreetMap', zoom_start = 7)

#  map the locations in Washington
for lat, lng, label in zip(sm.Latitude, sm.Longitude, sm.Location):
    folium.CircleMarker(
        [lat, lng],
        radius=1,
        popup=label,
        fill=True,
        color='Blue',
        fill_color='Yellow',
        fill_opacity=0.6
        ).add_to(map_world)

map_world
print("\nUNIQUE CLASSIFIERS")

# fill any missing data with the word "missing"
df = df.fillna('missing')

print("Geocode Status")
print(df["Geocode Status"].unique())

# The information about geocode status is from the Colorado data marketplace:
# https://data.colorado.gov/Energy/Alternative-Fuels-and-Electric-Vehicle-Charging-St/team-3ugz
# It is a rating indicating the approximate accuracy of the latitude and longitude for the station's address, 
# given as code values: 
#    GPS = The location is from a real GPS readout at the station. 
#    200-9 = Premise (building name, property name, shopping center, etc.) level accuracy. 
#    200-8 = Address level accuracy. 
#    200-7 = Intersection level accuracy. 
#    200-6 = Street level accuracy. 
#    200-5 = ZIP code (postal code) level accuracy. 
#    200-4 = Town (city, village) level accuracy. 
#    200-3 = Sub-region (county, municipality, etc.) level accuracy. 
#    200-2 = Region (state, province, prefecture, etc.) level accuracy. 
#    200-1 = Country level accuracy. 
#    200-0 = Unknown accuracy.
# This data could be used to finetune the script to add a note about the accuracy of the location

# in order to charge you must have the correct EV connector type for your electric vehicle

print("\nEV Connector Types\n")
print(df["EV Connector Types"].unique())
numJ1772 = df["EV Connector Types"].str.contains('J1772').value_counts().tolist()
print("\n",numJ1772[0]," stations have EV connector type:  J1772\n")
EVcon = df["EV Connector Types"].value_counts().copy()
for index, val in EVcon.iteritems():
    print(val, " stations have ", index)
# in order to charge you will need to use your EV network along with the proper access codes

print("\nAccess Code")
print(df["Access Code"].unique())

print("\nGroups With Access Code")
print(df["Groups With Access Code"].unique())

print("\nAccess Detail Code")
print(df["Access Detail Code"].unique())

print("\nEV Network")
print(df["EV Network"].unique())

print("\nCards Accepted")
print(df["Cards Accepted"].unique())
# many stations offer free charging and others charge a fee

print("\nEV Pricing")
df['EV Pricing'] = df['EV Pricing'].str.upper()
EVprice = df["EV Pricing"].value_counts().copy()
for index, val in EVprice.head(5).iteritems():
    print(val, " stations have pricing set at ", index)
# Before taking a trip, it is important to know whether the data about charging stations is current and accurate

print("\nDATA CONFIRMATION")
old = df.loc[~df["Date Last Confirmed"].str.contains('2019')].copy()
old.sort_values(by=["Date Last Confirmed"],inplace=True)
print(old.shape[0], " stations were confirmed BEFORE 2019 between ", end="")
print(old['Date Last Confirmed'].iloc[0], " and ", end="")
print(old['Date Last Confirmed'].iloc[-1])
print("\nMISSING CARD DATA")
#print(df["Cards Accepted"].isnull().sum())
print(df["Cards Accepted"].str.contains('missing').sum(), " stations are missing the cards they accept")
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