import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib.pyplot as plt
df = pd.read_csv ('../input/countries of the world.csv')
df.head()
df['Coastline (coast/area ratio)'] = df['Coastline (coast/area ratio)'].str.replace(',','.')
df['Agriculture']  =df['Agriculture'].str.replace(',','.')
df['Industry']  = df['Industry'].str.replace(',','.')
df['Service']  = df['Service'].str.replace(',','.')
df.head()
df_ll = df[df['Coastline (coast/area ratio)']=='0.00']
print(df_ll.shape)
print(df_ll['Region'].value_counts())
df_ll = df_ll.sort_values(by='GDP ($ per capita)', axis=0, ascending=False, kind='quicksort', na_position='last')
df_ll.head(n=5)
df_non_ll = df[df['Coastline (coast/area ratio)']!='0.00']

df_non_ll = df_non_ll.sort_values(by='GDP ($ per capita)', axis=0, ascending=False, kind='quicksort', na_position='last')
df_non_ll.head(n=5)
df_ll = df_ll.sort_values(by='GDP ($ per capita)', axis=0, ascending=False, kind='quicksort', na_position='last')
df_ll.tail(n=5)
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
    'a': df_ll['Agriculture'],
    'b': df_ll['Industry'],
    'c': df_ll['Service'],
    'text': df_ll['Country'],
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
      'text': 'Ternary Plot with Markers for agriculture, industry and service, for land locked countries',
        'x': 0.5,
        'y': 1.3,
        'font': { 'size': 15 }
    }]
}

fig = {'data': data, 'layout': layout}
py.iplot(fig, validate=False)
df_non_ll = df[df['Coastline (coast/area ratio)']!='0.00']
print(df_non_ll.head())
print(df_non_ll.shape)
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
    'a': df_non_ll['Agriculture'],
    'b': df_non_ll['Industry'],
    'c': df_non_ll['Service'],
    'text': df_non_ll['Country'],
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
      'text': 'Ternary Plot with Markers for agriculture, industry and service, for non land locked countries',
        'x': 0.5,
        'y': 1.3,
        'font': { 'size': 15 }
    }]
}

fig = {'data': data, 'layout': layout}
py.iplot(fig, validate=False)
df_ll.boxplot(column='GDP ($ per capita)')
plt.show();
df_non_ll.boxplot(column='GDP ($ per capita)')
plt.show();