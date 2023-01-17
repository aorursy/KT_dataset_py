# Imports

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import datetime



# Plotly

import chart_studio.plotly as py

import plotly.express as px

import plotly.graph_objects as go

import colorlover as cl

from plotly.subplots import make_subplots



# Colors

#maroon = rgb(128,0,0)

#chocolate = rgb(210,105,30)

#sky blue = rgb(135,206,235)

#olive drab = rgb(107,142,35)

#steel blue = rgb(70,130,180)
# Read the data

data = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv")

data.head(2)



# Make a copy of the data for future graphs

copy = data.copy()



# Missing values

data.isna().sum() # there are 1200 missing values for Max, Min and Land&Ocean Average Temp

plt.figure(figsize = (16, 3))

sns.heatmap(data.isna())
# Missing data with listwise deletion

# Because data is missing in chucks and we are dealing with time series data, we will delete all rows that have at least 

# one missing value.



data.dropna(axis = 0, inplace = True)





# Dealing with the DATE

data['Date'] = pd.to_datetime(data.dt) # converted all dates to the same format



data2 = data.copy() # create a new dataset

data2.drop(columns = ['dt'], axis = 1, inplace = True) # drop the dt column



# Creating new features

data2['day'] = data2['Date'].dt.day

data2['week'] = data2['Date'].dt.week

data2['month'] = data2['Date'].dt.month

data2['year'] = data2['Date'].dt.year



# Week data is not evenly distributed

data2['week'].value_counts() # very uneven information on weeks



# For future analysis, we will work only on yearly data, as average (because there are dates missing and data is not consistent)

earth_data = data2.groupby(by = 'year')[['LandAverageTemperature', 'LandAverageTemperatureUncertainty',

       'LandMaxTemperature', 'LandMaxTemperatureUncertainty',

       'LandMinTemperature', 'LandMinTemperatureUncertainty',

       'LandAndOceanAverageTemperature',

       'LandAndOceanAverageTemperatureUncertainty']].mean().reset_index()



earth_data['turnpoint'] = np.where(earth_data['year'] <= 1975, 'before', 'after') # creating a new columns

earth_data.head(2)
# Simple Summary Statistics

earth_data[['LandAverageTemperature', 'LandMaxTemperature', 

       'LandMinTemperature', 'LandAndOceanAverageTemperature']].describe()
# Creating the dataset - using copy

copy['Date'] = pd.to_datetime(copy.dt)

copy['year'] = copy['Date'].dt.year

land_avg = copy.groupby('year')['LandAverageTemperature', 'LandAverageTemperatureUncertainty'].mean().reset_index()



# Creating the graph

fig = go.Figure()

fig.update_layout(title="Land Average Temperature: 1750-2010", title_font_size = 20,

                  font=dict( family="Courier New, monospace", size=12,color="#7f7f7f"),

                  template = "ggplot2", hovermode= 'closest')

fig.update_xaxes(showline=True, linewidth=1, linecolor='gray')

fig.update_yaxes(showline=True, linewidth=1, linecolor='gray')



fig.add_trace(go.Scatter(x = land_avg['year'], y = land_avg['LandAverageTemperature'], mode = 'lines',

                        name = 'Land Avg Temp', marker_color='rgb(128, 0, 0)'))



fig.add_trace(go.Scatter(x = land_avg['year'], y = land_avg['LandAverageTemperatureUncertainty'], mode = 'lines',

                        name = 'Land Avg Temp Error', marker_color = 'rgb(107,142,35)'))
# Figure layout

fig = make_subplots(rows=2, cols=2, insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.3}])

fig.update_layout(title="When Global Warming Started?",font=dict( family="Courier New, monospace", size=12,color="#7f7f7f"),

                 template = "ggplot2", title_font_size = 20, hovermode= 'closest')

fig.update_xaxes(showline=True, linewidth=1, linecolor='gray')

fig.update_yaxes(showline=True, linewidth=1, linecolor='gray')



# Figure data

fig.add_trace(go.Scatter(x = earth_data['year'], y = earth_data['LandAverageTemperature'], mode = 'lines',

                        name = 'Land Avg Temp', marker_color='rgb(128, 0, 0)'), row = 1, col = 1)

fig.add_trace(go.Scatter( x=[1975, 1975], y=[7.5, 10], mode="lines",line=go.scatter.Line(color="gray"), showlegend=False),

             row = 1, col = 1)

#=============================================================================

fig.add_trace(go.Scatter(x = earth_data['year'], y = earth_data['LandMinTemperature'], mode = 'lines',

                        name = 'Land Min Temp', marker_color='rgb(210,105,30)'), row = 1, col = 2)

fig.add_trace(go.Scatter( x=[1975, 1975], y=[1.5, 4.5], mode="lines",line=go.scatter.Line(color="gray"), showlegend=False),

             row = 1, col = 2)

#=============================================================================

fig.add_trace(go.Scatter(x = earth_data['year'], y = earth_data['LandMaxTemperature'], mode = 'lines',

                        name = 'Land Max Temp', marker_color='rgb(135,206,235)'), row = 2, col = 1)

fig.add_trace(go.Scatter( x=[1975, 1975], y=[13, 15.5], mode="lines",line=go.scatter.Line(color="gray"), showlegend=False),

             row = 2, col = 1)

#=============================================================================

fig.add_trace(go.Scatter(x = earth_data['year'], y = earth_data['LandAndOceanAverageTemperature'], mode = 'lines',

                        name = 'Land&Ocean Avg Temp', marker_color='rgb(107,142,35)'), row = 2, col = 2)

fig.add_trace(go.Scatter( x=[1975, 1975], y=[14.5, 16], mode="lines",line=go.scatter.Line(color="gray"), showlegend=False),

             row = 2, col = 2)
# Change in average temperature before/after 1975



# Figure layout

fig = make_subplots(rows=2, cols=2, insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.3}])

fig.update_layout(title="Average Temperatures before and after 1975",font=dict( family="Courier New, monospace", size=12,color="#7f7f7f"),

                 template = "ggplot2", title_font_size = 20, hovermode= 'closest')



# Figure data

fig.add_trace(go.Box(x = earth_data['LandAverageTemperature'], y = earth_data['turnpoint'],boxpoints = 'all',jitter = 0.3, 

                     pointpos = -1.6, marker_color = 'rgb(128, 0, 0)', boxmean = True, name = 'Land Avg Temp'),

             row = 1, col = 1)

fig.add_trace(go.Box(x = earth_data['LandMinTemperature'], y = earth_data['turnpoint'],boxpoints = 'all',jitter = 0.3, 

                     pointpos = -1.6, marker_color = 'rgb(210,105,30)', boxmean = True, name = 'Land Min Temp'),

             row = 1, col = 2)

fig.add_trace(go.Box(x = earth_data['LandMaxTemperature'], y = earth_data['turnpoint'],boxpoints = 'all',jitter = 0.3, 

                     pointpos = -1.6, marker_color = 'rgb(135,206,235)', boxmean = True, name = 'Land Max Temp'),

             row = 2, col = 1)

fig.add_trace(go.Box(x = earth_data['LandAndOceanAverageTemperature'], y = earth_data['turnpoint'], boxpoints = 'all',jitter = 0.3, 

                     pointpos = -1.6, marker_color = 'rgb(107,142,35)', boxmean = True, name = 'Land&Ocean Avg Temp'),

             row = 2, col = 2)





fig.update_traces(orientation='h')
# Read the file (countries + cities)

countries = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv")



# Because the file is very big and there are many dates missing (like the last file), we will group by year

# create column year

countries['Date'] = pd.to_datetime(countries['dt'])

countries['year'] = countries['Date'].dt.year



# Group by year

by_year = countries.groupby(by = ['year', 'City', 'Country', 'Latitude', 'Longitude']).mean().reset_index()



# Append the continent & iso codes

continent_map = pd.read_csv("../input/country-mapping-iso-continent-region/continents2.csv")

continent_map['Country'] = continent_map['name']

continent_map = continent_map[['Country', 'region', 'alpha-2', 'alpha-3']]



# Add information

data = pd.merge(left = by_year, right = continent_map, on = 'Country', how = 'left')



# Filter starting 1825 - because some countries weren't monitored before this year on some periods, 

# the mean overall could be quite misleading (example: Americas have an increase from 1821 to 1825 of 5 points in temperature,

# but this happens only because in 1824 data for South America started to be collected)

data = data[data['year'] >= 1825]



# Datasets:



region = data.dropna(axis = 0).groupby(by = ['region', 'year']).mean().reset_index()

countries = data.dropna(axis = 0).groupby(by = ['region', 'Country', 'year']).mean().reset_index()

cities = data.dropna(axis = 0).groupby(by = ['region', 'Country', 'City', 'year', 'Latitude', 'Longitude']).mean().reset_index()
# Figure layout

fig = make_subplots(rows=1, cols=2, insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.3}])

fig.update_layout(title="Continents increase in Average Temperature", title_font_size = 20,

                  font=dict( family="Courier New, monospace", size=12,color="#7f7f7f"),

                  template = "ggplot2", hovermode= 'closest')

fig.update_xaxes(showline=True, linewidth=1, linecolor='gray')

fig.update_yaxes(showline=True, linewidth=1, linecolor='gray')



#============================= Scatter =============================

fig.add_trace(go.Scatter(x = region[region['region'] == 'Europe']['year'], y = region[region['region'] == 'Europe']['AverageTemperature'], mode = 'lines',

                        name = 'Europe', marker_color='rgb(128, 0, 0)'), row = 1, col = 1)

fig.add_trace(go.Scatter(x = region[region['region'] == 'Americas']['year'], y = region[region['region'] == 'Americas']['AverageTemperature'], mode = 'lines',

                        name = 'Americas', marker_color='rgb(210,105,30)'), row = 1, col = 1)

fig.add_trace(go.Scatter(x = region[region['region'] == 'Asia']['year'], y = region[region['region'] == 'Asia']['AverageTemperature'], mode = 'lines',

                        name = 'Asia', marker_color='rgb(135,206,235)'), row = 1, col = 1)

fig.add_trace(go.Scatter(x = region[region['region'] == 'Africa']['year'], y = region[region['region'] == 'Africa']['AverageTemperature'], mode = 'lines',

                        name = 'Africa', marker_color='rgb(107,142,35)'), row = 1, col = 1)

fig.add_trace(go.Scatter(x = region[region['region'] == 'Oceania']['year'], y = region[region['region'] == 'Oceania']['AverageTemperature'], mode = 'lines',

                        name = 'Oceania', marker_color='rgb(70,130,180)'), row = 1, col = 1)



#============================= Bar =============================

y1 = np.round(region.groupby(by = 'region')['AverageTemperature'].mean().tolist(), 1)

y2 = np.round(region.groupby(by = 'region')['AverageTemperature'].max().tolist(), 1)



fig.add_trace(go.Bar(x = region['region'].unique(), y = region.groupby(by = 'region')['AverageTemperature'].mean().tolist(), 

                     name = 'Mean Temp', marker_color = 'rgb(188,143,143)', text = y1, textposition = 'auto'),

              row = 1, col = 2)

fig.add_trace(go.Bar(x = region['region'].unique(), y = region.groupby(by = 'region')['AverageTemperature'].max().tolist(), 

                     name = 'Max Temp', marker_color = 'rgb(222,184,135)', text = y2, textposition = 'auto'),

              row = 1, col = 2)
# Data

mean = countries.groupby(['Country', 'region'])['AverageTemperature'].mean().reset_index()

maximum = countries.groupby(['Country', 'region'])['AverageTemperature'].max().reset_index()



difference = pd.merge(left = mean, right = maximum, on = ['Country', 'region'])

difference['diff'] = difference['AverageTemperature_y'] - difference['AverageTemperature_x']



# Graph

fig = go.Figure()

fig.update_layout(title="Difference in Temperature (Countries)", title_font_size = 20,

                  font=dict( family="Courier New, monospace", size=13,color="#7f7f7f"),

                  template = "ggplot2", autosize = False, height = 3500, width = 750)

fig.update_xaxes(showline=True, linewidth=1, linecolor='gray')

fig.update_yaxes(showline=True, linewidth=1, linecolor='gray')



sort_diff = difference[['Country', 'region', 'diff']].sort_values(by = 'diff', ascending = True)

fig.add_trace(go.Bar(x = sort_diff['diff'], y = sort_diff['Country'], orientation = 'h',

                    marker=dict(color='rgb(222,184,135)', line=dict( color='rgb(188,143,143)', width=0.6))))

fig.show()
# Data - we need iso alpha-3 codes

map_countries = data.dropna(axis = 0).groupby(by = ['region', 'Country', 'year','alpha-3']).mean().reset_index()



# Min temperature is -5.453083, and because the size in a map cannot be negative, we will add 6 to all temperatures

# to "standardize the data"

map_countries['AverageTemperature'] = map_countries['AverageTemperature'] + 6



fig = px.scatter_geo(map_countries, locations='alpha-3', color='region',

                     color_discrete_sequence = ('rgb(128,0,0)','rgb(210,105,30)','rgb(135,206,235)','rgb(107,142,35)'),

                     hover_name="Country", size="AverageTemperature", size_max=15, opacity = 0.8,

                     animation_frame="year",

                     projection="natural earth", title='Interactive Globe Map - Temperature increase')

fig.show()
# Calculating the difference column

mean = map_countries.groupby(['region','Country','alpha-3'])['AverageTemperature'].mean().reset_index()

maximum = map_countries.groupby(['region','Country','alpha-3'])['AverageTemperature'].max().reset_index()

difference = pd.merge(left = mean, right = maximum, on = ['region','Country','alpha-3'])

difference['diff'] = difference['AverageTemperature_y'] - difference['AverageTemperature_x']

difference.rename(columns = {'AverageTemperature_y':'Maximum Average Temperature',

                             'AverageTemperature_y':'Overall Avg Temp'}, inplace = True)



# Figure

fig = px.scatter_geo(difference, locations="alpha-3", color="Overall Avg Temp", #we color by average temp

                     hover_name="Country", size="diff", size_max=15, #we size by how big is the difference

                     projection="natural earth", opacity = 0.8,

                     color_continuous_scale=('#283747', '#2874A6', '#3498DB', '#F5B041', '#E67E22', '#A93226'),

                     title = 'Global Map - difference between the mean and max temperatures')

fig.show()