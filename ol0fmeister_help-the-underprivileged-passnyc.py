import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import folium

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
import plotly.plotly as py
SE = pd.read_csv("../input/data-science-for-good/2016 School Explorer.csv")
reg_test = pd.read_csv("../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv")
fin_emp_cent = pd.read_csv("../input/nyc-financial-empowerment-centers/financial-empowerment-centers.csv")
pd.set_option('display.max_columns', None) 
SE.head(3)
reg_test.head()
CORDS = (40.767937, -73.982155)
map_1 = folium.Map(location=CORDS, zoom_start=13)
folium.Marker([40.721834, -73.978766], popup='Roberto Clemente').add_to(map_1)
folium.Marker([40.729892, -73.984231], popup='Asher Levy').add_to(map_1)
folium.Marker([40.721274, -73.986315], popup='Anna Silver').add_to(map_1)
folium.Marker([40.726147, -73.975043], popup='Franklin D. Roosevelt').add_to(map_1)
folium.Marker([40.724404, -73.986360], popup='The Star Academy').add_to(map_1)
display(map_1)
SE['School Income Estimate'] = SE['School Income Estimate'].str.replace(',', '')
SE['School Income Estimate'] = SE['School Income Estimate'].str.replace('$', '')
SE['School Income Estimate'] = SE['School Income Estimate'].str.replace(' ', '')
SE['School Income Estimate'] = SE['School Income Estimate'].astype(float)
data = [
    {
        'x': SE["Longitude"],
        'y': SE["Latitude"],
        'text': SE["School Name"],
        'mode': 'markers',
        'marker': {
            'color': SE["Economic Need Index"],
            'size': SE["School Income Estimate"]/4500,
            'showscale': True,
            'colorscale':'Portland'
        }
    }
]

layout= go.Layout(
    title= 'New York School Population based on Economic Need Index',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='NYC_ECNEED_INDEX')
fin_emp_cent.head()
data = [
    {
        'x': fin_emp_cent["Longitude"],
        'y': fin_emp_cent["Latitude"],
        'text': fin_emp_cent["Provider"],
        'mode': 'markers',
        'marker': {
            'showscale': False,
            'colorscale':'Jet',
            'size': 20
        }
    }
]

layout= go.Layout(
    title= 'Location of Different Financial Empowerment Centres in NYC',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='FC_LOCATIONS')
data = [
    {
        'x': SE["Longitude"],
        'y': SE["Latitude"],
        'text': SE["School Name"],
        'mode': 'markers',
        'marker': {
            'color': SE["School Income Estimate"],
            'showscale': True,
            'colorscale':'Jet',
            'size': 10
        }
    }
]

layout= go.Layout(
    title= 'Location of NYC Schools based on their Income Estimates',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='NYC_INCOME')
plt.figure(figsize=(12,14))
ax=plt.subplot(211)
sns.boxplot(y=SE['School Income Estimate'],x=SE["District"])
ax.set_title('District vs Income Estimate')

plt.figure(figsize=(12,14))
ax=plt.subplot(211)
sns.boxplot(y=SE['School Income Estimate'],x=SE["City"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('City vs Income Estimate')
data = [
    {
        'x': SE["Longitude"],
        'y': SE["Latitude"],
        'text': SE["School Name"],
        'mode': 'markers',
        'marker': {
            'color': SE["Economic Need Index"],
            'size': SE["Grade 3 ELA 4s - Economically Disadvantaged"],
            'showscale': True,
            'colorscale':'Portland'
        }
    }
]

layout= go.Layout(
    title= 'Distribution of economically challenged students across NYC schools',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='NYC_ECNEED_INDEX')


