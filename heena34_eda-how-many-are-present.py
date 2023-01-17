import numpy as np
import pandas as pd 
import copy
import datetime as dt
import os
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()
import missingno as msn

import warnings
warnings.filterwarnings('ignore')
school = pd.read_csv('../input/2016 School Explorer.csv')
shsat_data = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
school.head()
shsat_data.head()
school.info(verbose=True, null_counts=True)
school['School Income Estimate'] = school['School Income Estimate']\
                                    .str.replace('$', '')\
                                    .str.replace(',','')\
                                    .astype(float)
trace0 = {
    'x': school['Longitude'],
    'y': school['Latitude'],
    'text': school['School Income Estimate'],
    'mode': 'markers',
    'markers': {
        'size': school['School Income Estimate'],
        'showscale': True,
        'colorscale': 'Portland'
    }
}

layout= go.Layout(
    title= 'New York School Economic Estimate',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    )
)
trace = [trace0]
fig = dict( data=trace, layout=layout )
py.iplot( fig, validate=False)
trace0 = {
    'x': school['Longitude'],
    'y': school['Latitude'],
    'text': school['Economic Need Index'],
    'mode': 'markers',
    'markers': {
        'size': school['Economic Need Index'],
        'color': school['Economic Need Index'],
        'showscale': True,
        'colorscale': 'Portland'
    }
}

layout= go.Layout(
    title= 'New York School Economic Need Index',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    )
)
trace = [trace0]
fig = dict( data=trace, layout=layout )
py.iplot( fig, validate=False)
attend_less_than30 = school[school['Student Attendance Rate'].str.replace('%', '').astype(float) <= 30]

trace0 = {
    'x': attend_less_than30['Longitude'],
    'y': attend_less_than30['Latitude'],
    'text': attend_less_than30['Student Attendance Rate'],
    'mode': 'markers',
    'marker': {
        'color': school["Economic Need Index"],
        'size': school["School Income Estimate"]/4000,
        'showscale': True,
        'colorscale':'Portland'
    }
}

layout= go.Layout(
    title= 'New York School Economic Need Index',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    )
)
trace = [trace0]
fig = dict( data=trace, layout=layout )
py.iplot( fig, validate=False)
school['Percent ELL'] = school['Percent ELL'].str.replace('%', '').astype(float)
school['Percent Asian'] = school['Percent Asian'].str.replace('%', '').astype(float)
school['Percent Hispanic'] = school['Percent Hispanic'].str.replace('%', '').astype(float)
school['Percent Black'] = school['Percent Black'].str.replace('%', '').astype(float)
school['Percent Black / Hispanic'] = school['Percent Black / Hispanic'].str.replace('%', '').astype(float)
school['Percent White'] = school['Percent White'].str.replace('%', '').astype(float)
trace0 = go.Bar(
    x = ['ELL', 'Asian', 'Hispanic', 'Black', 'Black/Hispanic', 'White'],
    y = [school['Percent ELL'].mean(), school['Percent Asian'].mean(), school['Percent Hispanic'].mean(),
         school['Percent Black'].mean(), school['Percent Black / Hispanic'].mean(), school['Percent White'].mean()]
)

layout= go.Layout(
    title= 'Percentage of Different Groups of People',
    xaxis= dict(
        title= 'Groups'
    ),
    yaxis=dict(
        title='Percentage'
    )
)
trace = [trace0]
fig = dict( data=trace, layout=layout )
py.iplot( fig, validate=False)

school.head()