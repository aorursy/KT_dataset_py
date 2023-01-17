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



fourteeners = pd.read_csv('../input/14er.csv',encoding='ISO-8859-1')

fourteeners=fourteeners[['Mountain Peak','Mountain Range','Elevation_ft','Difficulty','Distance_mi','Elevation Gain_ft','Lat','Long']]

fourteeners.columns = ['Peak','Range','Elevation','Difficulty','Distance (Miles)','Gain (Feet)','Latitude','Longitude']

#fourteeners.head()
custom_colormap = folium.StepColormap(['yellow','orange','red','black'], 

                         vmin=0, vmax=24,

                         index=[0,5,10,24],

                        caption='Distance (Miles)')

fourteeners = fourteeners.sort_values(by='Distance (Miles)', ascending=1)

fourteeners = fourteeners.reset_index()

pd.set_option('display.max_colwidth', 400)

fourteeners[['Peak','Elevation','Difficulty','Distance (Miles)','Gain (Feet)']].head(60)
temperature_map = color_coded_map(fourteeners, 39.965411,-105.293598,9,5,'Latitude','Longitude','Distance (Miles)','Peak',True,1,)

temperature_map
data= fourteeners

selectedColumn = 'Range'

title='Locations of 14ers in Colorado'

yAxisTitle = title

plotMultipleChoice(selectedColumn,data,title,yAxisTitle)
data= fourteeners

selectedColumn = 'Difficulty'

title='Difficulty Class Ratings of 14ers in Colorado'

yAxisTitle = title

plotMultipleChoice(selectedColumn,data,title,yAxisTitle)