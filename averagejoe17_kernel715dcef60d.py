# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.subplots import make_subplots # Plotting library (subplots)
import plotly.graph_objects as go # General purpose plotting library
from datetime import datetime # Base Datetime functionality

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
measurement_summary = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')
measurement_info = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_info.csv')
measurement_item_info = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')
measurement_station_info = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')

measurement_summary['Measurement date'] = pd.to_datetime(measurement_summary['Measurement date'])
for i in [measurement_summary, measurement_info, measurement_item_info, measurement_station_info]:
    print(i.head())
    print(i.shape)
# Is pollution getting worse over time?
pollution_data_over_time = \
 (
     measurement_summary.groupby('Measurement date')                           # Group by datetime
                        .mean()                                                # Find the average
                        .drop(['Station code','Latitude','Longitude'],axis=1)  # Drop columns where average doesnt make sense
                        .unstack(level=-1)                                     # Unpivot pollutants
                        .reset_index()                                         # Remove columns from index
 ) 
pollution_data_over_time.columns = ['Pollutant','Date','PollutantLevel']                   # Relabel
pollution_data_over_time = pollution_data_over_time[['Date','Pollutant','PollutantLevel']] # Reorder
pollution_data_over_time                                                                   # Review

fig = make_subplots(rows=3, cols=2 , subplot_titles=pollution_data_over_time['Pollutant'].unique())

x, y = (0,1)
for i,p in enumerate(pollution_data_over_time['Pollutant'].unique()):
    print(i,p)
    
    # Change Subplots
    if i%2 == 1:
        y = 2
    else:
        y = 1
        x += 1
                
    # Fill Subplot
    fig.add_trace(
        go.Scatter(x=pollution_data_over_time[pollution_data_over_time['Pollutant']==p]['Date'],
                   y=pollution_data_over_time[pollution_data_over_time['Pollutant']==p]['PollutantLevel']),
                   row=x,
                   col=y)
fig.show()
import plotly.graph_objects as go

fig = go.Figure(go.Scattergeo(
                    lon = measurement_summary[measurement_summary['Measurement date'] ==datetime(day=1,month=1,year=2017,hour=1)]['Longitude'],
                    lat = measurement_summary[measurement_summary['Measurement date'] ==datetime(day=1,month=1,year=2017,hour=1)]['Latitude'],
                    hoverinfo = 'text',
                    text = measurement_summary[measurement_summary['Measurement date'] ==datetime(day=1,month=1,year=2017,hour=1)]['Station code'],
                    mode = 'markers',
                    marker = dict(
                                size = measurement_summary[measurement_summary['Measurement date'] ==datetime(day=1,month=1,year=2017,hour=1)]['PM10'],
                                color = 'rgb(255, 0, 0)'
                             )
                )
        )
fig.update_geos(projection_type="natural earth")
fig.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
fig.show()