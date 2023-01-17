import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.offline as py
from plotly import graph_objs as go
import folium
py.init_notebook_mode(False)

import os
input_path = '../input/gtd/globalterrorismdb_0718dist.csv'

data = pd.read_csv(input_path, encoding = 'ISO-8859-1', low_memory = False)
data.head()
uniques = data.nunique()
missing = data.isnull().sum()
trace = go.Scatter(
    x = uniques.index,
    y = uniques.values / data.shape[0] * 100,
    mode = 'markers',
    name = 'Unique %',
    marker = dict(
        #size = uniques.values / data.shape[0] * 100,
        sizemode = 'area',
        color = np.random.randn(len(uniques))
    )
)

trace1 = go.Scatter(
    x = missing.index,
    y = missing.values / data.shape[0] * 100,
    mode = 'markers',
    name = 'Missing %',
    marker = dict(
        #size = missing.values / data.shape[0] * 100,
        sizemode = 'area',
        color = np.random.randn(len(missing)),
        opacity = 0.5
    )
)

layout = go.Layout(
    title = 'Distinct Feature Information',
    xaxis = dict(
        title = 'Feature Names'
    ),
    yaxis = dict(
        title = 'Percentage of Values'
    )
)

fig = go.Figure(data = [trace, trace1], layout = layout)
py.iplot(fig)
data['Casualities'] = data['nkill'].fillna(0) + data['nwound'].fillna(0)
countries_affected = data.groupby(['country_txt'])['Casualities'].sum().sort_values(ascending = False)

colors = ['rgb(100, 30, 22)', 'rgb(120, 40, 31)', 'rgb(123, 36, 28)','rgb(148, 49, 38)', 'rgb(146, 43, 33)', 'rgb(176, 58, 46)',
          'rgb(169, 50, 38)', 'rgb(203, 67, 53)', 'rgb(192, 57, 43)', 'rgb(231, 76, 60)', 'rgb(205, 97, 85)', 'rgb(236, 112, 99)',
          'rgb(217, 136, 128)', 'rgb(241, 148, 138)', 'rgb(230, 176, 170)']

trace = go.Bar(
    x = countries_affected.index[:15],
    y = countries_affected.values[:15],
    marker = dict(
        color = colors,
        line = dict(
            color = 'rgb(8,48,107)',
            width = 1.5,
        )
    )
)

layout = go.Layout(
    title = 'Country Devastation',
    xaxis = dict(
        title = 'Country'
    ),
    yaxis = dict(
        title = 'Casuality'
    )
)

fig = go.Figure(data = [trace], layout = layout)

py.iplot(fig)
temp = data.groupby(['iyear','country_txt'])['success'].sum()
temp = temp.loc[temp.groupby('iyear').idxmax()].reset_index()
traces = []
temp = temp.groupby('country_txt')
for group in temp:
    g = temp.get_group(group[0])
    trace = go.Scatter(
        x = g['iyear'],
        y = g['success'],
        mode = 'markers+lines',
        marker = dict(
            sizemode = 'area',
            size = g['success'] * 0.3
        ),
        name = g['country_txt'].values[0]
    )
    traces.append(trace)

layout = go.Layout(
    title = 'Most Attacks Operated by Country',
    xaxis = dict(
        title = 'Year'
    ),
    yaxis = dict(
        title = '# of Successful Attacks'
    )
)

fig = go.Figure(data = traces, layout = layout)

py.iplot(fig)
yearly_killed = data.groupby(['iyear'])['nkill'].sum().reset_index()
yearly_wounded = data.groupby(['iyear'])['nwound'].sum().reset_index()

trace = go.Bar(
    x = yearly_killed['iyear'],
    y = yearly_killed['nkill'],
    name = 'Killed',
    marker = dict(
        color = 'red'
    )
)

trace1 = go.Bar(
    x = yearly_wounded['iyear'],
    y = yearly_wounded['nwound'],
    name = 'Wounded',
    marker = dict(
        color = 'red',
        opacity = 0.5
    )
    
)

layout = go.Layout(
    title = 'Yearly Casualities',
    xaxis = dict(
        title = 'Year'
    ),
    barmode = 'stack'
)

fig = go.Figure(data = [trace, trace1], layout = layout)
py.iplot(fig)
months = {'1': 'Jan', '2': 'Feb', '3': 'March', '4': 'April', '5': 'May', '6': 'June', '7': 'July', '8': 'Aug', '9': 'Sept', '10': 'Oct', 
          '11': 'Nov', '12': 'Dec'}

monthly_deaths = data.groupby(['imonth'])['nkill'].sum()

#traces = []
#for year in monthly_deaths.index.levels[0]:
trace = go.Scatter(
    x = [months[str(i)] for i in monthly_deaths.index.values[1:]],
    y = monthly_deaths.values[1:],
    #name = year,
    line = dict(
        color = 'rgb(23, 32, 42)'
    ),
    fill = 'toself',
    fillcolor = 'red',
    opacity = 0.6
)

layout = go.Layout(
    title = 'Killings Per Month',
    xaxis = dict(
        title = 'Month'
    ),
    yaxis = dict(
        title = 'Kill Count'
    )
)

fig = go.Figure(data = [trace], layout = layout)
    
py.iplot(fig)
group_names = data['gname'].value_counts()[1:11]

trace = go.Bar(
    x = group_names.index,
    y = group_names.values,
    
    marker = dict(
        color = colors[:10],
        line = dict(
            color='rgb(8,48,107)',
                    width = 1.5
        ),
    )
)

layout = go.Layout(
    title = 'Notorious Terrorist Groups',
    xaxis = dict(
        title = 'Terrorist Group'
    ),
    yaxis = dict(
        title = '# Attacks'
    )
)

fig = go.Figure(data = [trace], layout = layout)

py.iplot(fig)
def get_Fre(data, col_name):
    data = data.groupby(['gname', col_name])[col_name].count().unstack().T
    x_axis = data.columns
    y_axis = np.max(data, axis = 0)
    text = data.idxmax().values
    return x_axis, y_axis, text

def plot(xaxis, yaxis, text):
    u = np.unique(text)
    color_num = np.random.randn(len(u)) * 0.2
    colors = []
    for t1 in text:
        for index, t2 in enumerate(u):
            if t1 == t2:
                colors.append(color_num[index])

    trace = go.Bar(
        x = xaxis,
        y = yaxis,
        text = text,
        textposition = 'outside',
        hoverinfo = 'text + x',
        marker = dict(
            color = colors,
            line = dict(
                color = 'rgb(8,48,107)',
                width = 1.5,
            )
        ),
    )
    
    return trace

notorious_groups = data[data['gname'].isin(group_names.index)][['gname', 'attacktype1_txt', 'targtype1_txt', 'weaptype1_txt']]

trace1 = plot(*get_Fre(notorious_groups, 'targtype1_txt'))
trace2 = plot(*get_Fre(notorious_groups, 'attacktype1_txt'))
trace3 = plot(*get_Fre(notorious_groups, 'weaptype1_txt'))
from plotly import tools
fig = tools.make_subplots(rows = 3, cols = 1, shared_xaxes = True, print_grid = False, subplot_titles = ('Terrorist Group vs. Terrorist Target',
                                                                                                         'Terrorist Group vs. Attack Type',
                                                                                                         'Terrorist Group vs. Weapon Type'))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)

fig['layout'].update(height = 600, width = 1000, showlegend = False)
py.iplot(fig)
casualities = data.groupby('country_txt')['Casualities'].sum().reset_index().sort_values(by = 'Casualities', ascending = False)

m = folium.Map(
    location = [12, 12],
    zoom_start = 2,
    tiles = 'CartoDB positron'
)

m.choropleth(
    geo_data = os.path.join('../input/worldcountries', 'world-countries.json'),
    data = casualities,
    columns = casualities.columns,
    key_on = 'feature.properties.name',
    fill_color = 'YlOrRd',
    line_opacity = 0.5,
    fill_opacity = 0.8,
    reset = True,
    smooth_factor = 1.0
    
)
m