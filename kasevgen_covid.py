import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timedelta

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/covid19-full-dataset/owid-covid-data (1).csv')
df
# viridis_cmap = matplotlib.cm.get_cmap('viridis')

# viridis_rgb = []
# norm = matplotlib.colors.Normalize(vmin=0, vmax=255)

# for i in range(0, 255):
#     k = matplotlib.colors.colorConverter.to_rgb(viridis_cmap(norm(i)))
#     viridis_rgb.append(f'rgb{k}')
def dynamic_bar_country(df, country, count_hist=10, shift=50, count_ma=20):
    data = df[df['location'] == country].iloc[:, 3:10]
    data.set_index(pd.DatetimeIndex(data['date']), inplace=True)
    data['ExpMA'] = data['new_cases'].ewm(span=count_ma, adjust=False).mean()

    listOfFrames = []

    for j, i in enumerate(pd.date_range(data.index[0], data.index[-count_hist])):
        v = pd.date_range(i, periods=count_hist).to_series()
#         print(v)
        d1 = data.loc[v, 'new_cases']
        d2 = data.loc[v, 'new_deaths']
        d3 = data.loc[v, 'ExpMA']

        data1 = go.Bar(x=v, y=d1, name='Infected', 
                       marker_color=[f'rgb{k + shift, shift, shift}' for k in range(j, j + count_hist)],
                       hoverinfo="none", textposition="outside", texttemplate="%{y:s}", cliponaxis=False)

        data2 = go.Bar(x=v, y=d2, name='Died', 
                       marker_color=[f'rgb{shift, k + shift, shift}' for k in range(j, j + count_hist)],
                       hoverinfo="none", textposition="outside", texttemplate="%{y:s}", cliponaxis=False)
        
        data3 = go.Scatter(x=v, y=d3, name='Exp MA', mode="lines")
        
        layout = go.Layout(font={"size": 14}, height=400, width=600,
                           barmode='group', bargap=0.15, bargroupgap=0.1,
                           xaxis={"showline": True, "tickangle": 30, "visible": True, "linewidth": 2, "linecolor": 'black'},
                           yaxis={"showline": True, "tickangle": 30, "visible": True, "linewidth": 2, "linecolor": 'black'},
                           title=str(datetime.strptime(str(v[-1]), '%Y-%m-%d %H:%M:%S').date()))

        listOfFrames.append(go.Frame(data=[data1, data2, data3], layout=layout))
    
    d1 = go.Bar(x=data.index, y=data['new_cases'], name='Infected', 
                marker={'color': data['new_cases'], 'colorscale': 'Viridis'}, hoverinfo="none", 
                textposition="outside", texttemplate="%{y:s}", cliponaxis=False)
    
    d2 = go.Bar(x=data.index, y=data['new_deaths'], name='Died',
                marker={'color': data['new_deaths'], 'colorscale': 'Magma'}, hoverinfo="none", 
                textposition="outside", texttemplate="%{y:s}", cliponaxis=False)
    
    d3 = go.Scatter(x=data.index, y=data['ExpMA'], name='Exp MA', mode="lines")
    
    layout = go.Layout(title=f'Covid in {country}', font={"size": 16}, height=400, width=600,
                       barmode='group', bargap=0.15, bargroupgap=0.1,
                       xaxis={"showline": True, "tickangle": 30, "visible": True, "linewidth": 2, "linecolor": 'black'},
                       yaxis={"showline": True, "tickangle": 30, "visible": True, "linewidth": 2, "linecolor": 'black'},
                       updatemenus=[dict(type="buttons",
                                         buttons=[dict(label="►", method="animate", args=[None, {"fromcurrent": True}]),
                                                  dict(label="❚❚", method="animate", args=[[None], {"frame": {"duration": 200, "redraw": False}, 
                                                                       "mode": 'immediate', "transition": {"duration": 200}}])])])
    fig = go.Figure(data=[d1, d2, d3], layout=layout, frames=list(listOfFrames))
#     print(list(listOfFrames))
    return fig
fig1 = dynamic_bar_country(df, 'United States')
fig1.show()
fig1 = dynamic_bar_country(df, 'Italy')
fig1.show()
df2 = df[df['location'] == 'Italy'].iloc[:, 3:10]
df2.set_index(pd.DatetimeIndex(df2['date']), inplace=True)
df2.drop(['date'], axis=1, inplace=True)

df2['new_cases'].min(), df2['new_deaths'].min()
import plotly.express as px

list_country = ['United States', 'Russia', 'Brazil', 'India', 'Italy', 
                'Spain', 'China', 'Sweden', 'United Kingdom', 'Sweden']

df2 = df[df['continent'].notna()]
df2 = df2[df2['location'].isin(list_country)].iloc[:, :8]
df2['date'] = pd.to_datetime(df2['date'])
df2['total_cases'] = df2['total_cases'].astype(int)


fig = px.bar(df2, x='location', y='total_cases', height=400, width=600, 
             color="location", template="plotly_dark",
             animation_frame=pd.DatetimeIndex(df2['date']).strftime('%Y-%m-%d'),
             animation_group="location", log_y=True, title='Increased infections Covid')
fig.show()
list_del = []
for i in list_country:
    diff = pd.date_range(start='2019-12-31', end='2020-07-09').difference(df2[df2['location'] == i]['date'])
    print(i, diff, sep='\t')
data = df[df['continent'].notna()].iloc[:, :8]
max_value = data[data['date'] == '2020-07-09']['total_cases'].quantile(.99)
data = data[data['date'] > '2020-03-15']
fig = px.choropleth(data, locations="iso_code", height=400, width=600, 
                    title='Increased infections Covid in world',
                    color="total_cases", template="plotly_dark", hover_name="location", 
                    color_continuous_scale=px.colors.sequential.Plasma, 
                    range_color=(0, max_value),
                    animation_frame=pd.DatetimeIndex(data['date']).strftime('%Y-%m-%d'))

fig.show()
import plotly as py
py.offline.init_notebook_mode(connected=True)

py.offline.plot(fig, filename='myplot.html')
