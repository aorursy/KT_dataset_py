import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib.pyplot as plt

df = pd.read_csv("../input/countries of the world.csv", decimal=',')
#print(df.head())
metricscale1=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'], 
              [0.15, 'rgb(171,221,164)'], [0.2, 'rgb(230,245,152)'], 
              [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'], 
              [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = metricscale1,
        showscale = True,
        locations = df['Country'].values,
        z = df['GDP ($ per capita)'].values,
        locationmode = 'country names',
        text = df['Country'].values,
        marker = dict(
            line = dict(color = 'rgb(250,250,225)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'GDP')
            )
       ]

layout = dict(
    title = 'World Map of GDP ($US per capita)',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(28,107,160)',
        #oceancolor = 'rgb(222,243,246)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmapGDP')
metricscale1=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'], 
              [0.15, 'rgb(171,221,164)'], [0.2, 'rgb(230,245,152)'], 
              [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'], 
              [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = metricscale1,
        showscale = True,
        locations = df['Country'].values,
        z = df['Infant mortality (per 1000 births)'].values,
        locationmode = 'country names',
        text = df['Country'].values,
        marker = dict(
            line = dict(color = 'rgb(250,250,225)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Infant Mort.')
            )
       ]

layout = dict(
    title = 'World Map of Infant mortality (per 1000 births)',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(28,107,160)',
        #oceancolor = 'rgb(222,243,246)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmapInfantMortality')
df.plot(x='GDP ($ per capita)',y='Infant mortality (per 1000 births)',kind='scatter')
plt.show()
df.plot(x='GDP ($ per capita)',y='Infant mortality (per 1000 births)',kind='scatter',loglog=True)
plt.show()
def makeAxis(title, tickangle): 
    return {
      'title': title,
      'titlefont': { 'size': 20 },
      'tickangle': tickangle,
      'tickfont': { 'size': 15 },
      'tickcolor': 'rgba(0,0,0,0)',
      'ticklen': 5,
      'showline': True,
      'showgrid': True
    }

data = [{ 
    'type': 'scatterternary',
    'mode': 'markers',
    'a': df['Agriculture'],
    'b': df['Industry'],
    'c': df['Service'],
    'text': df['Country'],
    'marker': {
        'symbol': 100,
        'color': '#DB7365',
        'size': 14,
        'line': { 'width': 2 }
    },
    }]

layout = {
    'ternary': {
        'sum': 1,
        'aaxis': makeAxis('Agriculture', 0),
        'baxis': makeAxis('<br>Industry', 45),
        'caxis': makeAxis('<br>Service', -45)
    },
    'annotations': [{
      'showarrow': False,
      'text': 'Ternary Plot with Markers for agriculture, industry and service',
        'x': 0.5,
        'y': 1.3,
        'font': { 'size': 15 }
    }]
}

fig = {'data': data, 'layout': layout}
py.iplot(fig, validate=False)