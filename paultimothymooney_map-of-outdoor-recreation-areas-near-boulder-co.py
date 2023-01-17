import numpy as np

import pandas as pd

import os

import folium

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)





def plotPointsOnMap(dataframe,beginIndex,endIndex,latitudeColumn,latitudeValue,longitudeColumn,longitudeValue,zoom):

    df = dataframe[beginIndex:endIndex]

    location = [latitudeValue,longitudeValue]

    plot = folium.Map(location=location,zoom_start=zoom)

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.name[i:i+1]))

        folium.Marker([df[latitudeColumn].iloc[i],df[longitudeColumn].iloc[i]],popup=popup).add_to(plot)

    return(plot)



def color_coded_map(df, center_lat, center_lon, zoom,bubbleSize,latitudeColumn,longitudeColumn,coloredColumn,labeledColumn,fillBool,opacityValue):  

    # Adapted from  https://www.kaggle.com/dejavu23/openaq-from-queries-to-world-maps

    m = folium.Map(location=[center_lat, center_lon],

                   zoom_start = zoom)

    for index, row in df.iterrows():

        folium.CircleMarker([row[[latitudeColumn]], row[[longitudeColumn]]],radius=bubbleSize, 

                             color=custom_colormap(row[coloredColumn]), #

                             fill_color=custom_colormap(row[coloredColumn]), #

                             fill=fillBool, fill_opacity=opacityValue,             

                             popup=row[[labeledColumn]]).add_to(m) 

    custom_colormap.add_to(m)

    folium.TileLayer(tiles='Stamen Toner',name="Stamen Toner").add_to(m)

    folium.TileLayer(tiles='Stamen Terrain',name="Stamen Terrain").add_to(m)

    folium.TileLayer(tiles='cartodbpositron',name="cartodbpositron").add_to(m)

    folium.LayerControl().add_to(m)       

    return m



def plotMultipleChoice(selectedColumn,data,title,yAxisTitle):

    counts = data[selectedColumn].value_counts()

    countsDf = pd.DataFrame(counts)

    trace1 = go.Bar(

                    x = countsDf.index,

                    y = countsDf[selectedColumn],

                    name = "Kaggle",

                    marker = dict(color = 'rgba(0, 0, 255, 0.8)',

                                 line=dict(color='rgb(0,0,0)',width=1.5)),

                    text = countsDf.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=title, yaxis= dict(title=yAxisTitle),showlegend=False)

    fig = go.Figure(data = data, layout = layout)

    iplot(fig)



OSMP_Trails = pd.read_csv('../input/OSMP_Trails.csv')

OSMP_Trails=OSMP_Trails[['TRAILNAME','TRAILTYPE','BICYCLES','HORSES','MILEAGE']]

OSMP_Trailheads = pd.read_csv('../input/OSMP_Trailheads.csv')

OSMP_Trailheads=OSMP_Trailheads[['OSMPTrailheadsOSMPTHNAME','OSMPTrailheadsOSMPADDRESS','OSMPTrailheadsOSMPELEVATION','OSMPTrailheadsOSMPPARKSPACES',

                                 'OSMPTrailheadsOSMPRESTROOMS','OSMPTrailheadsOSMPBIKERACK','OSMPTrailheadsOSMPBIKETRAIL','X','Y']]

OSMP_Climbing_Formations = pd.read_csv('../input/OSMP_Climbing_Formations.csv')

OSMP_Climbing_Formations=OSMP_Climbing_Formations[['FEATURE','AreaAccess','ROUTES','UseRating','FormationType','X','Y']]
custom_colormap = folium.StepColormap(['yellow','orange','red','black'], 

                         vmin=0, vmax=4,

                         index=[0,1,2,3],

                        caption='User Rating')

OSMP_Climbing_Formations = OSMP_Climbing_Formations.sort_values(by='UseRating', ascending=0)

OSMP_Climbing_Formations.head(60)
temperature_map = color_coded_map(OSMP_Climbing_Formations, 39.965411,-105.293598,12,5,'Y','X','UseRating','FEATURE',True,1,)

temperature_map
data= OSMP_Climbing_Formations

selectedColumn = 'FormationType'

title='Types of Climbing Formations near Boulder'

yAxisTitle = title

plotMultipleChoice(selectedColumn,data,title,yAxisTitle)
data= OSMP_Climbing_Formations

selectedColumn = 'AreaAccess'

title='Number of Climbing Formations at Each Location'

yAxisTitle = title

plotMultipleChoice(selectedColumn,data,title,yAxisTitle)
custom_colormap = folium.StepColormap( ['purple','blue','green','yellow','orange','red'], 

                         vmin=5000, vmax=10000,

                         index=[5000,6000,7000,8000,9000,10000],

                        caption='Elevation')

OSMP_Trailheads = OSMP_Trailheads.sort_values(by='OSMPTrailheadsOSMPELEVATION', ascending=0)

OSMP_Trailheads.head(20)
temperature_map = color_coded_map(OSMP_Trailheads, 39.997336,-105.294556, 12,5,'Y','X','OSMPTrailheadsOSMPELEVATION','OSMPTrailheadsOSMPTHNAME',True,1,)

temperature_map
data= OSMP_Trails

selectedColumn = 'TRAILTYPE'

title='Types of Hiking Trails near Boulder'

yAxisTitle = title

plotMultipleChoice(selectedColumn,data,title,yAxisTitle)
data= OSMP_Trails

selectedColumn = 'HORSES'

title='Are Horses Allowed?'

yAxisTitle = title

plotMultipleChoice(selectedColumn,data,title,yAxisTitle)
data= OSMP_Trails

selectedColumn = 'BICYCLES'

title='Are Bicycles Allowed?'

yAxisTitle = title

plotMultipleChoice(selectedColumn,data,title,yAxisTitle)