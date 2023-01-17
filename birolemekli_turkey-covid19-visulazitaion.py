import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df=pd.read_csv('/kaggle/input/corona-virus-test-numbers-in-turkey/turkey_covid19_all.csv')
df.Confirmed=df.Confirmed.astype(int)
df.Recovered=df.Recovered.astype(int)
df.Deaths=df.Deaths.astype(int)
df['yeni_vaka']=df['Confirmed'] - df['Confirmed'].shift(1,fill_value=0)
df['yeni_iyilesen']=df['Recovered'] - df['Recovered'].shift(1,fill_value=0)
df['yeni_vefat']=df['Deaths']-df['Deaths'].shift(1,fill_value=0)
df.tail()
df.info()
df.describe()
df.drop(columns='Province/State',inplace=True)
maps = folium.Map(location=[39.951483, 32.857478],  zoom_start=4)
folium.Marker(
    location=[39.951483, 32.857478],
    icon=folium.Icon(color='red',prefix='fa'),
    tooltip='<bold>Vefat : '+str(round(df['Deaths'].max()))
).add_to(maps)
folium.Marker(
    location=[39.551483, 33.877478],
    icon=folium.Icon(color='green',prefix='fa'),
    tooltip='<bold>Iyilesen : '+str(round(df['Recovered'].max()))
).add_to(maps)
folium.Marker(
    location=[39.951483, 34.997478],
    icon=folium.Icon(color='blue',prefix='fa'),
    tooltip='<bold>Vaka : '+str(round(df['Confirmed'].max()))
).add_to(maps)
maps
fig = go.Figure(data=go.Scatter(
    x=df.Date,
    y=df.Confirmed,
    mode='markers',
    marker_color='rgba(0, 0, 255, .7)',
    marker=dict(size=[x for x in range(5,len(df.Confirmed)+6,1)],
                color=[x for x in range(len(df.Confirmed))])
))
fig.update_layout(
    title="Vaka Durumu",
    xaxis_title="Tarih",
    yaxis_title="Sayı"
)
fig.show()
fig = go.Figure(data=go.Scatter(
    x=df.Date,
    y=df.Deaths,
    mode='lines+markers',
    marker_color='rgba(255, 0, 0, .9)',
    marker_line_width=2, marker_size=10
))
fig.update_layout(
    title="Vefat Durumu",
    xaxis_title="Tarih",
    yaxis_title="Sayı"
)
fig.show()
fig = go.Figure(data=go.Scatter(
    x=df.Date,
    y=df.Recovered,
    mode='lines+markers',
    marker_color='green',
    marker_line_width=2, marker_size=10
))
fig.update_layout(
    title="İyileşen Durumu",
    xaxis_title="Tarih",
    yaxis_title="Sayı"
)
fig.show()
labels=['Hasta','İyileşen','Vefat']
values=[df.Confirmed.max()-df.Recovered.max(),df.Recovered.max(),df.Deaths.max()]
irises_colors = ['rgb(0, 0, 255)', 'rgb(50, 200, 110)', 'rgb(255, 0, 0)']
fig = make_subplots(1, specs=[[{'type':'domain'}]],subplot_titles=['Türkiye'])
fig.add_trace(go.Pie(labels=labels, values=values, pull=[0,0,0.1], hole=.4,marker_colors=irises_colors))
fig.update_layout(title_text='Türkiye Vaka Durumu')
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.Date, y=df.Confirmed,
                    marker_color='blue',
                    mode='markers',
                    name='Hasta'))
fig.add_trace(go.Scatter(x=df.Date, y=df.Recovered,
                    marker_color='green',
                    mode='lines+markers',
                    name='İyileşen'))
fig.add_trace(go.Scatter(x=df.Date, y=df.Deaths,
                    marker_color='red',
                    mode='lines',
                    name='Vefat'))
fig.update_layout(title='Covid 19 Yayılım Hızı',
                  yaxis_zeroline=False, xaxis_zeroline=False)
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.Date, y=df.yeni_vaka,
                    marker_color='blue',
                    mode='lines',
                    name='Hasta'))
fig.add_trace(go.Scatter(x=df.Date, y=df.yeni_iyilesen,
                    marker_color='green',
                    mode='lines+markers',
                    name='İyileşen'))
fig.update_layout(title='Günlük Hasta İyileşen Tablosu',
                  yaxis_zeroline=False, xaxis_zeroline=False)
fig.show()
df.isna().sum()
plt.figure(figsize=(8, 6))
ax=sns.heatmap(df[['Tests','yeni_vaka','yeni_iyilesen','yeni_vefat']].corr(), 
               annot=True,
               cbar = True,  
               square = True,
               annot_kws={'size': 10},
               cmap= 'coolwarm')
years = df.Date
fig = go.Figure()
fig.add_trace(go.Bar(x=years,
                y=df.yeni_vaka,
                name='Vaka',
                marker_color='rgb(0, 0, 255)'
                ))
fig.add_trace(go.Bar(x=years,
                y=df.yeni_iyilesen,
                name='İyileşen',
                marker_color='rgb(0,255, 0)'
                ))
fig.add_trace(go.Bar(x=years,
                y=df.yeni_vefat,
                name='Vefat',
                marker_color='rgb(255, 0, 0)'
                ))
fig.update_layout(
    title='Günlük Vaka Durumu',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Sayı',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.03,
    bargroupgap=0.1
)
fig.show()
years = df.Date
fig = go.Figure()
fig.add_trace(go.Bar(x=years,
                y=df.Confirmed,
                name='Vaka',
                marker_color='rgb(0, 0, 255)'
                ))
fig.add_trace(go.Bar(x=years,
                y=df.Recovered,
                name='İyileşen',
                marker_color='rgb(0,255, 0)'
                ))
fig.add_trace(go.Bar(x=years,
                y=df.Deaths,
                name='Vefat',
                marker_color='rgb(255, 0, 0)'
                ))
fig.update_layout(
    title='Toplam Vaka Durumu',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Sayı',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.03,
    bargroupgap=0.1
)
fig.show()
fig = go.Figure(data=[go.Mesh3d(x=df.Confirmed,
                   y=df.Recovered,
                   z=df.Deaths,
                   opacity=0.5,)])


fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),
                    width=700,
                    margin=dict(
                    r=10, l=10,
                    b=10, t=10)
                  )
fig.show()
