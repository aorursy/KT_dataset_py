# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
dfc=pd.read_csv('../input/countries/countries.csv')
dfcod=pd.read_csv('../input/codes-countries/2014_world_gdp_with_codes.csv')
dfcod.head()
data = [dict(
    type='scattergeo',
    lon = dfc['Longitude'],
    lat = dfc['Latitude'],
    hoverinfo = 'text',
    text = "Country: " + dfc.Country,
    mode = 'markers',
    marker=dict(
        sizemode = 'area',
        sizeref = 1,
        size= 10 ,
        symbol = 'circle',
        line = dict(width=1,color = "white"),
        colorscale='Viridis',
        opacity = 0.7),
)]
layout = dict(
    autosize=True,
    height=700,
    title = 'Countries Of the World',
    hovermode='closest',
    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,
               countrywidth=2, projection=dict(type='robinson'),
               landcolor = 'rgba(255, 255, 204, 1)',
               subunitwidth=2,
               showlakes = False,
               showocean=True,
               countrycolor="rgba(1, 50, 67, 1)")
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#['Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu',
#            'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet',
#            'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis']
        
data = [ dict(
        type = 'choropleth',
        locations = dfcod['CODE'],
        z = dfcod['GDP (BILLIONS)'],
        text = dfcod['COUNTRY'],
        zmin=0,zmax=5000,
        colorscale='Blues',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) )
      ) ]

layout = dict(
    autosize=True,
    height=750,
    title = 'Countries by GDP (BILLIONS)',
    geo = dict(
        showframe = True,
        showcoastlines = True,
        showocean=True,
        projection = dict(
            type = 'orthographic'
        )
    )
)

fig = dict(data=data, layout=layout)
iplot(fig)