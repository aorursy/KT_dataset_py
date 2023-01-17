# Visualisations
import matplotlib.pyplot as plt 
import matplotlib
import plotly
import plotly.offline as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly.tools import make_subplots
init_notebook_mode()

import seaborn as sns
plt.style.use('fivethirtyeight')
%matplotlib inline
# Warnings
import warnings
warnings.filterwarnings('ignore')

# Data exploration
import pandas as pd

# Numerical
import numpy as np

# Spatial analysis
import geopandas as gpd   # used to read .shp file
from shapely.geometry import Point, Polygon, shape
# import utm   # Need to be installed, coordinates conversion

# Regular expressions
import re

# Random
np.random.seed(11)

# Files in dataset
import os
print(os.listdir('../input/nyc-crimes-2018-random-forest-regressor-nans'))   # 2nd kernel output
print(os.listdir('../input/ny-police-precincts'))   # Police precincts boundary folder
# Main df
crimes_original = pd.read_csv('../input/nyc-crimes-2018-random-forest-regressor-nans/crimes_NYC.csv')
crimes_original[:5]
# Precincts are not needed with this version of notebook. You can skip this cell.
# Precincts
precincts = gpd.read_file('../input/ny-police-precincts/geo_export_e973c2ae-18a9-437a-8f40-bf039d82ad2e.shp')
precincts[:5]
# Change CMPLNT_FR_DT series to datetime.time type
crimes_original.CMPLNT_FR_TM = pd.to_datetime(crimes_original.CMPLNT_FR_TM, format='%H:%M:%S').dt.time
type(crimes_original.CMPLNT_FR_TM[0])
# Choose colors for plotly plots
import colorlover as cl
from IPython.display import HTML

chosen_colors=cl.scales['9']['seq']['BuPu']
print('The color palette chosen for this notebook is:')
HTML(cl.to_html(chosen_colors))
# Counted
borogroup = crimes_original.BORO_NM.value_counts()
boro = borogroup.index
# Crimes divided into boroughs - plotly
trace1 = go.Bar(
    x = boro,
    y = borogroup,
    name = 'No. of crimes',
    textposition = 'outside',
    marker = dict(color=chosen_colors[6],
                line=dict(
                    color='rgb(48,12,80)',
                    width=2.5,
                )        
    ),
    opacity = 0.65
)

data=[trace1]

layout = go.Layout(
    title = 'No. of crimes in New York Boroughs',
    xaxis = dict( title = 'Borough'),
    yaxis = dict( title = 'Np. of crimes'), width=700, height=500)
   

figure = go.Figure(data=data, layout=layout)
py.iplot(figure)
# Calculate incidents dvided into level of offence
sum_mis = crimes_original.BORO_NM[crimes_original.LAW_CAT_CD == 'MISDEMEANOR'].value_counts()
sum_fel = crimes_original.BORO_NM[crimes_original.LAW_CAT_CD == 'FELONY'].value_counts()
sum_vio = crimes_original.BORO_NM[crimes_original.LAW_CAT_CD == 'VIOLATION'].value_counts()
# Grouped bar chart
trace1 = go.Bar(
    x = boro,
    y = sum_mis,
    name = 'MISDEMEANOR',
    marker = dict(color=chosen_colors[7]),
    opacity = 0.7
)

trace2 = go.Bar(
    x = boro,
    y = sum_fel,
    name = 'FELONY',
    marker = dict(color=chosen_colors[5]),
    opacity = 0.7
)

trace3 = go.Bar(
    x = boro,
    y = sum_vio,
    name = 'VIOLATION',
    marker = dict(color=chosen_colors[3]),
    opacity = 0.7
)

data = [trace1, trace2, trace3]
layout = go.Layout(
    title = 'No. of incidents in New York Boroughs divided into levels of offence',
    barmode='group',
    xaxis = dict( title = 'Borough'),
    yaxis = dict( title = 'Np. of crimes'), width=700, height=500
)

figure = go.Figure(data=data, layout=layout)
py.iplot(figure)
# Prepare data for for horizontal bar chart

# time boundaries
five_am = pd.Timestamp('5:00').time()
one_pm = pd.Timestamp('13:00').time()
nine_pm = pd.Timestamp('21:00').time()
midnight = pd.Timestamp('00:00').time()

#Calculating x_data
sum_5_13 = crimes_original.BORO_NM[(crimes_original.CMPLNT_FR_TM > five_am)
                                   & (crimes_original.CMPLNT_FR_TM <= one_pm)].value_counts()
sum_13_21 = crimes_original.BORO_NM[(crimes_original.CMPLNT_FR_TM > one_pm)
                                   & (crimes_original.CMPLNT_FR_TM <= nine_pm)].value_counts()
sum_21_5 = crimes_original.BORO_NM[((crimes_original.CMPLNT_FR_TM > nine_pm)
                                   & (crimes_original.CMPLNT_FR_TM <= midnight))
                                   | ((crimes_original.CMPLNT_FR_TM > midnight) 
                                   & (crimes_original.CMPLNT_FR_TM <= five_am))].value_counts()

# x_data dataframe
x_df = pd.DataFrame({'sum_5_13': sum_5_13, 'sum_13_21': sum_13_21, 'sum_21_5': sum_21_5})
x_df['sum_5_13%'] = (x_df['sum_5_13'] / x_df.sum(axis=1)).mul(100)
x_df['sum_13_21%'] = (x_df['sum_13_21'] / x_df.sum(axis=1)).mul(100)
x_df['sum_21_5%'] = (x_df['sum_21_5'] / x_df.sum(axis=1)).mul(100)
x_df_percent = x_df[['sum_5_13%', 'sum_13_21%', 'sum_21_5%']]

# I had to follow example from plot.ly site and I converted df values to list of lists.
def make(x_df_percent):
    brooklyn=[]
    manhattan=[]
    bronx=[]
    queens=[]
    staten_island=[]
    all_to = [brooklyn, manhattan, bronx, queens, staten_island]
    for i in range(0, len(all_to)):
        for j in range(0, x_df_percent.shape[1]):
            all_to[i].append(x_df_percent.iloc[i, j])
            
    return all_to

all_to = make(x_df_percent)
# Horizontal bar plot with percents of crimes in time of day

top_labels = ['Early mornig,<br>midday', 'Afternoon,<br>early evening', 'Late evening,<br>night']

x_data = all_to
y_data = x_df_percent.index

traces = []
# Plot traces - 15 elements
for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            traces.append(go.Bar(
                x=[xd[i]],
                y=[yd],
                orientation='h',
                marker=dict(
                    color=chosen_colors[7-i],
                    line=dict(
                        color='rgb(248, 248, 255)',
                        width=1)
                ),
            ))

layout = go.Layout(
    title='Percentage share of NYC boroughs crimes at different times',
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        domain=[0.15, 1]
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
    
    ),
    barmode='stack',
    bargap=0.3,
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    margin=dict(
        l=120,
        r=10,
        t=140,
        b=80
    ),
    showlegend=False,
    height = 450,
    width = 700
)

annotations = []

for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref='paper', yref='y',
                            x=0.11, y=yd,
                            xanchor='right',
                            text=str(yd),
                            font=dict(family='Arial', size=16,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False, align='right'))
    # labeling the first percentage of each bar (x_axis)
    annotations.append(dict(xref='x', yref='y',
                            x=xd[0] / 2, y=yd,
                            text=str(xd[0].round(1)) + '%',
                            font=dict(family='Arial', size=14,
                                      color='rgb(248, 248, 255)'),
                            showarrow=False))
    # labeling the first position on the top
    if yd == y_data[-1]:
        annotations.append(dict(xref='x', yref='paper',
                                x=xd[0] / 2, y=1.18,
                                text=top_labels[0],
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False))
    space = xd[0]
    
    for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=space + (xd[i]/2), y=yd, 
                                    text=str(xd[i].round(1)) + '%',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the top scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i]/2), y=1.18,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]

layout['annotations'] = annotations
            
fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
# Violin plot, crimes divided into days of week

# new column in df - day of week
crimes_original['inc_day'] =  pd.to_datetime(crimes_original.CMPLNT_FR_DT,  format='%m/%d/%Y', errors='coerce').dt.dayofweek
# round incident time to fullhour
crimes_original.CMPLNT_FR_TM = crimes_original.CMPLNT_FR_TM.apply(lambda x: x.hour)
# map with dict
days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
crimes_original['inc_day'] = crimes_original['inc_day'].map(days)
# above solution is not perfect, bot works and I`ll improve it in near future
crimes_original['inc_day'] = crimes_original['inc_day'].fillna(method='ffill')

# All works perfect technically, but methond of filling nans is inappropriate and I don`t know why changing string to datetime 
# doesn`t work with date like 01/01/2017. Solve it! 

# chart layout params
vals = [1, 4, 7, 10, 13, 16, 19, 22]   # y-axis values
# order weekdays
array = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
crimes_original['inc_day'] = pd.Categorical(crimes_original['inc_day'], categories=array, ordered=True)
crimes_original.sort_values(by='inc_day', inplace=True)

#plot
data = []
for i in range(0, len(pd.unique(crimes_original['inc_day']))):
    trace = {
        'type': 'violin',
        'x': crimes_original['inc_day'][crimes_original['inc_day'] == pd.unique(crimes_original['inc_day'])[i]],
        'y': crimes_original['CMPLNT_FR_TM'][crimes_original['inc_day'] == pd.unique(crimes_original['inc_day'])[i]],
        'name': pd.unique(crimes_original['inc_day'])[i],
        'box': {
            'visible': True
        },
        'meanline': {
            'visible': True
        },
        "color": chosen_colors[i]
    }
    data.append(trace)
    
fig = {
    'data': data,
    'layout': {
        'title': 'Total incident distribution<br>',
        'yaxis': {
            'zeroline': False,
            'tickvals': vals,
        },
        'height': 700,
    }
}

py.iplot(fig, validate=False)
# Violin plot suspector gender in weekdays

showlegend = [True, False, False, False, False, False, False]

data = []
for i in range(0, len(pd.unique(crimes_original['inc_day']))):
    male = {
            "type": 'violin',
            'x': crimes_original['inc_day'][(crimes_original['suspector_sex_rand'] == 'M') & (crimes_original['inc_day'] == pd.unique(crimes_original['inc_day'])[i])],
            'y': crimes_original['CMPLNT_FR_TM'][(crimes_original['suspector_sex_rand'] == 'M') & (crimes_original['inc_day'] == pd.unique(crimes_original['inc_day'])[i])],
            "legendgroup": 'M',
            "scalegroup": 'M',
            "name": 'M',
            "side": 'negative',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": '#C9DB6E'
            },
            "marker": {
                "line": {
                    "width": 2,
                    "color": '#C9DB6E'
                }
            },
        'showlegend': showlegend[i] 
        }
    data.append(male)
    female = {
            "type": 'violin',
            'x': crimes_original['inc_day'][(crimes_original['suspector_sex_rand'] == 'F') & (crimes_original['inc_day'] == pd.unique(crimes_original['inc_day'])[i])],
            'y': crimes_original['CMPLNT_FR_TM'][(crimes_original['suspector_sex_rand'] == 'F') & (crimes_original['inc_day'] == pd.unique(crimes_original['inc_day'])[i])],
            "legendgroup": 'F',
            "scalegroup": 'F',
            "name": 'F',
            "side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": '#F5845B'
            },
            "marker": {
                "line": {
                    "width": 2,
                    "color": '#F5845B'
                }
            },
        'showlegend': showlegend[i]
        }
    data.append(female)

fig = {
    "data": data,
    "layout" : {
        "title": "Incidents distribution<br>scaled by number of incidents per gender",
        "yaxis": {
            "zeroline": False,
            "tickvals": vals,
            "title": "Hours"
        },
        "violingap": 0,
        "violingroupgap": 0,
        "violinmode": "overlay",
        'height': 700
    }
}


py.iplot(fig, validate = False)
# Incidents by hour
incidents_hour = crimes_original.groupby(['CMPLNT_FR_TM', 'suspector_sex_rand']).count().reset_index()

trace = go.Bar(
    x=incidents_hour.CMPLNT_FR_TM,
    y=incidents_hour.ADDR_PCT_CD,
    name='Number of incidents',
    marker=dict(
        color=chosen_colors[6],
        opacity=0.65,
    )
)

data = [trace]

layout = go.Layout(
    title='No. of incidents by hour',
    xaxis=dict(
        title='Hour',
        tick0=0,
        dtick=1,
    ),
    yaxis=dict(
        title='Number of incidents'
    ),
    barmode='stack',
    height=500,
    width=1000, 
)

figure = go.Figure(data=data, layout=layout)

py.iplot(figure)
# Stacked bar plot
temp_df = incidents_hour

data = []

traces_names = ['M', 'F']

for i in range(2):
    temp_df1=temp_df[(incidents_hour.suspector_sex_rand==traces_names[i])]
    data.append(
        go.Bar(
            x = temp_df1.CMPLNT_FR_TM,
            y = temp_df1.ADDR_PCT_CD,
            name = traces_names[i],
            marker = dict(
                color=chosen_colors[5-i]
            )
        )
    )
    
layout = go.Layout(
    title='No. of crimes per hour in NYC',
    xaxis=dict(
        title='Hour',
        tick0=0,
        dtick=1,
    
    ),
    yaxis=dict(
        title='No. of crimes'
    ),
    barmode='stack',
    height=500,
    width=1000
)

figure = go.Figure(data=data, layout=layout)

py.iplot(figure)
# Access to mapbox maps
# Every user should have own public access token. Please, don't use below. You can easily generate your own here: https://www.mapbox.com/
# If you use this token and I change it your map won't be working.
mapbox_access_token = 'pk.eyJ1IjoibWloYWx3MjgiLCJhIjoiY2psejZqZThnMXRndDNxcDFpdWh6YnV2NCJ9.IGbFZyg0dcy61geuwJUByw'
# Map using mapbox access token and plotly

mapbox_access_token  # https://www.mapbox.com/

# new variables
crimes_lon = crimes_original.Longitude
crimes_lat = crimes_original.Latitude

data = [
    go.Scattermapbox(
        lat=crimes_lat[:5000], 
        lon=crimes_lon[:5000],
        mode='markers',
        marker=dict(
            size=5,
            color='rgb(155, 132, 204)',
            opacity=0.5
        ),
        text=['New York'],
        hoverinfo='none'
    ),
    go.Scattermapbox(
        lat=crimes_lat[:10000],
        lon=crimes_lon[:10000],
        mode='markers',
        marker=dict(
            size=3,
            color='rgb(155, 132, 204)',
            opacity=0.7
        ),
        hoverinfo='none'
    )
]

layout = go.Layout(
    title="New York Crimes Locations",
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=40.729302,
            lon=-73.986670
        ),
        pitch=45,
        zoom=13.18,
        style='mapbox://styles/mihalw28/cjlz6bzrv697i2rpec2yqartr',# Inspiration and more interesting this type visualisations: https://www.kaggle.com/kosovanolexandr/crimes-in-boston-multiclass-clustering
    ),
    #width = 700,
    height = 850
)

fig = dict(data=data, layout=layout)

py.iplot(fig)
# Create a list with coordinates for folium
locations = crimes_original[['Latitude', 'Longitude']]
locationlist = locations.values.tolist()
# Mapping with folium - works good for 1k not for 100k

#Import maps
import folium
from folium import plugins

map_osm = folium.Map(location=[40.6865,-73.9496], tiles='Stamen Terrain', zoom_start=10)


marker_cluster = plugins.MarkerCluster().add_to(map_osm)

#I don`t know how to add a popup station`s name imported from df_stations['start_station_name'] column, ordinary solution doesn't work :/
# parse_html=True - this is the solution

for point in range(0, len(crimes_original[:1000])):
    folium.Marker(locationlist[point],
                  popup=folium.Popup(crimes_original['PATROL_BORO'][point], parse_html=True),
                  icon=folium.Icon(color='blue', icon_color='white', icon='fa-circle', angle=0, prefix='fa')).add_to(marker_cluster)

map_osm
