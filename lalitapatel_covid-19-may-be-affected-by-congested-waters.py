from __future__ import print_function

import os

import pandas as pd

import numpy as np

import math

import datetime as dt

import json

# import plotly.express as px

import folium

from urllib.request import urlopen

from IPython.core.display import display, HTML
zfLk = "../input/geofiles/ne_50m_lakes.geojson"

# zdir = "../input/outlier-areas/" {This did not work.}

zdir = "http://lipy.us/docs/OutlierAreas/"

zfCc = zdir + "WorldCovidCases.csv"

zfWr = zdir + "WuhanRoads.JPG"

zfWv = zdir + "WuhanWaters1.JPG"

zfWw = zdir + "WuhanWaters2.JPG"

zfIr = zdir + "ItalyRoads.JPG"

zfIv = zdir + "ItalyWaters1.JPG"

zfIw = zdir + "ItalyWaters2.JPG"

zfNr = zdir + "NewYorkRoads.JPG"

zfNv = zdir + "NewYorkWaters1.JPG"

zfNw = zdir + "NewYorkWaters2.JPG"

zfSr = zdir + "SeattleRoads.JPG"

zfSv = zdir + "SeattleWaters1.JPG"

zfSw = zdir + "SeattleWaters2.JPG"
df = pd.read_csv(zfCc, encoding='latin-1')

df = df.fillna("x")

df['Latitude'] = df['Latitude'].apply(lambda x: int(x))

df['Longitude'] = df['Longitude'].apply(lambda x: int(x))

df = df.groupby(by=['Latitude','Longitude']).agg({'Cases':'sum','Deaths':'sum','Country':'max','Code':'max','Capital':'max'})

df = df.sort_values(by=['Cases'], ascending=False)

df = df.reset_index()

df.head(5)
with open(zfLk) as resp:

    zjso = json.loads(resp.read())

    

zlls = []

for zloc in zjso['features']:

    zlls.append([zloc['properties']['name'],zloc['geometry']['coordinates'][0][0][0],zloc['geometry']['coordinates'][0][1][0]])

zlls = pd.DataFrame(data=zlls, columns=['name','lat','long'])

zlls.head(3)
zmap = folium.Map([20,0], zoom_start=2, tiles='cartodbpositron')



colorC = {0:'orange', 1:'beige', 2:'purple', 3:'darkpurple', 4:'lightred', 5:'red', 6:'darkred'}

# green,darkgreen,lightgreen,blue,darkblue,cadetblue,lightblue,white,pink,gray,black,lightgray



for Lat,Lon,Cas,Dea,Cou,Cap in zip(df['Latitude'],df['Longitude'],df['Cases'],df['Deaths'],df['Country'],df['Capital']):

    

    Val = int(math.log(Cas+9, 8))

    

    folium.CircleMarker(

        location = [Lat,Lon],

        radius = Val,

        popup = str(Cou) + '<br>' + str(Cap) + '<br>' +str(Cas) + '<br>' + str(Dea),

        threshold_scale = [0,1,2,3,4,5,6],

        color = colorC[Val],

        fill_color = colorC[Val],

        fill = False,

        fill_opacity = 1

    ).add_to(zmap)



folium.GeoJson(zjso).add_to(zmap)

# folium.LatLngPopup().add_to(zmap)



zmap
zimg = [[zfWr,zfWv,zfWw],[zfIr,zfIv,zfIw],[zfNr,zfNv,zfNw],[zfSr,zfSv,zfSw]]



zhtm = "<table><tr><td>.</td><td>Wuhan China</td><td>Lombardy Italy</td><td>New York USA</td><td>Seattle</td></tr>"

for i in range(3):

    zhtm = zhtm + "<tr><td>" + ["Roads","Sat Image","sat Image"][i] + "</td>"

    for j in range (4):

        zhtm = zhtm + "<td><img style='width:120px;height:180px;margin:0px;float:left;border:5px solid green;' src='" + zimg[j][i] + "' /></td>"

    zhtm = zhtm + "</tr>"

zhtm = zhtm + "</table>"

    

display(HTML(zhtm))