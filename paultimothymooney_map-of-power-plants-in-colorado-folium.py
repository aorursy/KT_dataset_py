import folium

import numpy as np

import pandas as pd

import os



baseDir = '../input/globalpowerplantdatabasev110/'

fileName = 'global_power_plant_database.csv'

filePath = os.path.join(baseDir,fileName)

everything = pd.read_csv(filePath)



def plotPointsOnMap(dataframe,beginIndex,endIndex,latitudeColumn,latitudeValue,longitudeColumn,longitudeValue,zoom):

    df = dataframe[beginIndex:endIndex]

    location = [latitudeValue,longitudeValue]

    plot = folium.Map(location=location,zoom_start=zoom)

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.fuel1[i:i+1]))

        folium.Marker([df[latitudeColumn].iloc[i],df[longitudeColumn].iloc[i]],popup=popup).add_to(plot)

    return(plot)
colorado_latitudeLower = everything['latitude'] > 36

colorado_latitudeUpper = everything['latitude'] < 42

colorado_longitudeLower = everything['longitude'] > -109

colorado_longitudeUpper = everything['longitude'] < -102

colorado_only = everything[colorado_latitudeLower & colorado_latitudeUpper & colorado_longitudeLower & colorado_longitudeUpper]

plotPointsOnMap(colorado_only,0,425,'latitude',39.7348,'longitude',-104.9653,6)