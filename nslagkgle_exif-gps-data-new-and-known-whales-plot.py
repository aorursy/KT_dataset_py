import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
plotly.offline.init_notebook_mode()
whales = pd.read_csv('../input/whalegps/train_gps_info.csv')
whales.head(2)
lng_plt = []
lat_plt = []
for i in range(len(whales)):
    # latitude
    if whales.latDMSRef.iloc[i]=="N":
        lat_plt+=[whales.latDD.iloc[i]]
    else:
        lat_plt+=[whales.latDD.iloc[i]* -1]
    
    # longitude
    if whales.lngDMSRef.iloc[i]=="E":
        lng_plt+=[whales.lngDD.iloc[i]]
    else:
        lng_plt+=[whales.lngDD.iloc[i]* -1] 
whales["lat"]=lat_plt
whales["lng"]=lng_plt
whales.head(2)
new_whales = whales[whales["Id"]=="new_whale"]
print(len(new_whales))
new_coords = pd.concat([new_whales['lat'], new_whales['lng'], new_whales['Id']], axis=1)
known_whales = whales[whales["Id"]!="new_whale"]
print(len(known_whales))
known_coords = pd.concat([known_whales['lat'], known_whales['lng'], known_whales['Id']], axis=1)
cases=[]

# known whales
cases.append(go.Scattergeo(
    lon = known_coords['lng'],
    lat = known_coords['lat'],
    mode = 'markers',
    marker = dict(
         size = 5,
         color = 'rgb(0,128,0)', # green
         opacity = 0.8,
         line = dict(width = 1.5))))
             
# new whales
cases.append(go.Scattergeo(
    lon = new_coords['lng'],
    lat = new_coords['lat'],
    mode = 'markers',
    marker = dict(
         size = 5,
         color = 'rgb(255,0,0)', # red 
         opacity = 0.8,
         line = dict(width = 1.5))))
layout = go.Layout(
    title = 'Humpback Whale - Exif data',
    autosize=False,
    width=900,
    height=700,
    margin=dict(
        t=30,
        b=10, 
        l=5, 
        r=5),
    geo = dict(
        resolution = 110,
        scope = 'world',
        showframe = True,
        showcoastlines = True,
        showland = True,
        landcolor = "rgb(229, 229, 229)",
        countrycolor = "rgb(255, 255, 255)" ,
        coastlinecolor = "rgb(255, 255, 255)",
        projection = dict(
            type = 'mercator'
        )))
fig = go.Figure(layout=layout, data=cases)
plotly.offline.iplot(fig, filename='whaleSpottings')