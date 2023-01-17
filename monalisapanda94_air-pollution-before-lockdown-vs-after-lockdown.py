# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
%matplotlib inline
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
#plt.rcParams['figure.figsize']=17,8
import cufflinks as cf
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
import folium
df=pd.read_csv('/kaggle/input/air-quality-data-in-india/city_day.csv')
df.head()
df.isnull().sum()
df  = df.fillna(df.mean())
df['BTX'] = df['Benzene']+df['Toluene']+df['Xylene']
df.drop(['Benzene','Toluene','Xylene'],axis=1,inplace=True)
df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d') # date parse
df['year'] = df['Date'].dt.year # year
df['year'] = df['year'].fillna(0.0).astype(int)
df['month'] = [d.strftime('%b') for d in df.Date]
S=df.groupby('City')['SO2'].max().sort_values(ascending=False).reset_index()
S.head()
trace = go.Table(
    domain=dict(x=[0, 0.52],
                y=[0, 1.0]),
    header=dict(values=["City","SO2"],
                fill = dict(color = '#119DFF'),
                font = dict(color = 'white', size = 14),
                align = ['center'],
               height = 30),
    cells=dict(values=[S['City'].head(10),S['SO2'].head(10)],
               fill = dict(color = ['lightgreen', 'white']),
               align = ['center']))

trace1 = go.Bar(x=S['City'].head(10),
                y=S['SO2'].head(10),
                xaxis='x1',
                yaxis='y1',
                marker=dict(color='lime'),opacity=0.60)
layout = dict(
    width=830,
    height=420,
    autosize=False,
    title='TOP 10 Cities with Max SO2',
    showlegend=False,   
    xaxis1=dict(**dict(domain=[0.58, 1], anchor='y1', showticklabels=True)),
    yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),  
)

fig1 = dict(data=[trace, trace1], layout=layout)
iplot(fig1)
SO2=df.groupby('year')['SO2'].sum().reset_index().sort_values(by='year',ascending=False)
NO2=df.groupby('year')['NO2'].sum().reset_index().sort_values(by='year',ascending=False)
BTX=df.groupby('year')['BTX'].sum().reset_index().sort_values(by='year',ascending=False)
CO=df.groupby('year')['CO'].sum().reset_index().sort_values(by='year',ascending=False)
PM=df.groupby('year')['PM2.5'].sum().reset_index().sort_values(by='year',ascending=False)
O=df.groupby('year')['O3'].sum().reset_index().sort_values(by='year',ascending=False)
!pip install chart_studio
plt.subplots(figsize =(15,8))
sns.pointplot(x='month', y='SO2', data=df,color='Orange')
plt.xlabel('MONTHS',fontsize = 16,color='blue')
plt.ylabel('SO2',fontsize = 16,color='blue')
plt.title('SO2 in Different Months',fontsize = 20,color='blue')
N=df.groupby('City')['NO2'].max().sort_values(ascending=False).reset_index()
N.head()
trace = go.Table(
    domain=dict(x=[0, 0.52],
                y=[0, 1.0]),
    header=dict(values=["City","NO2"],
                fill = dict(color = '#119DFF'),
                font = dict(color = 'white', size = 14),
                align = ['center'],
               height = 30),
    cells=dict(values=[N['City'].head(10),N['NO2'].head(10)],
               fill = dict(color = ['#25FEFD', 'white']),
               align = ['center']))

trace1 = go.Bar(x=N['City'].head(10),
                y=N['NO2'].head(10),
                xaxis='x1',
                yaxis='y1',
                marker=dict(color='darkblue'),opacity=0.60)
layout = dict(
    width=830,
    height=430,
    autosize=False,
    title='TOP 10 Cities with Max NO2',
    showlegend=False,   
    xaxis1=dict(**dict(domain=[0.58, 1], anchor='y1', showticklabels=True)),
    yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),  
)

fig1 = dict(data=[trace, trace1], layout=layout)
iplot(fig1)
plt.subplots(figsize =(15,8))
sns.pointplot(x='month', y='NO2', data=df,color='darkblue')
plt.xlabel('MONTHS',fontsize = 16,color='blue')
plt.ylabel('NO2',fontsize = 16,color='blue')
plt.title('NO2 in Different Months',fontsize = 20,color='blue')
C=df.groupby('City')['CO'].max().sort_values(ascending=False).reset_index()
C.head()
trace = go.Table(
    domain=dict(x=[0, 0.52],
                y=[0, 1.0]),
    header=dict(values=["City","CO"],
                fill = dict(color = 'red'),
                font = dict(color = 'white', size = 14),
                align = ['center'],
               height = 30),
    cells=dict(values=[C['City'].head(10),C['CO'].head(10)],
               fill = dict(color = ['lightsalmon', 'white']),
               align = ['center']))

trace1 = go.Bar(x=C['City'].head(10),
                y=C['CO'].head(10),
                xaxis='x1',
                yaxis='y1',
                marker=dict(color='fuchsia'),opacity=0.60)
layout = dict(
    width=830,
    height=490,
    autosize=False,
    title='TOP 10 Cities with Max CO',
    showlegend=False,   
    xaxis1=dict(**dict(domain=[0.58, 1], anchor='y1', showticklabels=True)),
    yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),  
)

fig1 = dict(data=[trace, trace1], layout=layout)
iplot(fig1)
plt.subplots(figsize =(15,8))
sns.pointplot(x='month', y='CO', data=df,color='green')
plt.xlabel('MONTHS',fontsize = 16,color='blue')
plt.ylabel('CO',fontsize = 16,color='blue')
plt.title('CO in Different Months',fontsize = 20,color='blue')
P=df.groupby('City')['PM2.5'].max().sort_values(ascending=False).reset_index()
P.head()
trace = go.Table(
    domain=dict(x=[0, 0.52],
                y=[0, 1.0]),
    header=dict(values=["City","PM2.5"],
                fill = dict(color = '#119DFF'),
                font = dict(color = 'white', size = 14),
                align = ['center'],
               height = 30),
    cells=dict(values=[P['City'].head(10),P['PM2.5'].head(10)],
               fill = dict(color = ['#25FEFD', 'white']),
               align = ['center']))

trace1 = go.Bar(x=P['City'].head(10),
                y=P['PM2.5'].head(10),
                xaxis='x1',
                yaxis='y1',
                marker=dict(color='deeppink'),opacity=0.60)
layout = dict(
    width=830,
    height=430,
    autosize=False,
    title='TOP 10 Cities with Max PM2.5',
    showlegend=False,   
    xaxis1=dict(**dict(domain=[0.58, 1], anchor='y1', showticklabels=True)),
    yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),  
)

fig1 = dict(data=[trace, trace1], layout=layout)
iplot(fig1)
plt.subplots(figsize =(15,8))
sns.pointplot(x='month', y='PM2.5', data=df,color='deeppink')
plt.xlabel('MONTHS',fontsize = 16,color='blue')
plt.ylabel('PM2.5',fontsize = 16,color='blue')
plt.title('PM2.5 in Different Months',fontsize = 20,color='blue')
B=df.groupby('City')['BTX'].max().sort_values(ascending=False).reset_index()
B.head()
trace = go.Table(
    domain=dict(x=[0, 0.52],
                y=[0, 1.0]),
    header=dict(values=["City","BTX"],
                fill = dict(color = '#119DFF'),
                font = dict(color = 'white', size = 14),
                align = ['center'],
               height = 30),
    cells=dict(values=[B['City'].head(10),B['BTX'].head(10)],
               fill = dict(color = ['#25FEFD', 'white']),
               align = ['center']))

trace1 = go.Bar(x=B['City'].head(10),
                y=B['BTX'].head(10),
                xaxis='x1',
                yaxis='y1',
                marker=dict(color='midnightblue'),opacity=0.60)
layout = dict(
    width=830,
    height=430,
    autosize=False,
    title='TOP 10 Cities with Max BTX',
    showlegend=False,   
    xaxis1=dict(**dict(domain=[0.58, 1], anchor='y1', showticklabels=True)),
    yaxis1=dict(**dict(domain=[0, 1.0], anchor='x1', hoverformat='.2f')),  
)

fig1 = dict(data=[trace, trace1], layout=layout)
iplot(fig1)
plt.subplots(figsize =(15,8))
sns.pointplot(x='month', y='BTX', data=df,color='salmon')
plt.xlabel('MONTHS',fontsize = 16,color='blue')
plt.ylabel('BTX',fontsize = 16,color='blue')
plt.title('BTX in Different Months',fontsize = 20,color='blue')
from plotly.tools import make_subplots
trace1=go.Scatter(x=SO2['year'], y=SO2['SO2'], mode='lines+markers', name='NO2')
trace2=go.Scatter(x=NO2['year'], y=NO2['NO2'], mode='lines+markers', name='NO2')
trace3=go.Scatter(x=CO['year'], y=CO['CO'], mode='lines+markers', name='CO')
trace4=go.Scatter(x=PM['year'], y=PM['PM2.5'], mode='lines+markers', name='PM2.5')
fig = plotly.tools.make_subplots(rows=2, cols=2,print_grid=False,
                          subplot_titles=('SO2 in diff. years','NO2 in diff. years','CO in diff. years',
                                          'PM2.5 in diff. years'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 2)
fig['layout'].update(height=550, width=850,title='AIR Pollutants In different Years',showlegend=False)
iplot(fig)
fig=go.Figure()
fig.add_trace(go.Scatter(x=SO2['year'], y=SO2['SO2'], mode='lines+markers', name='SO2',line=dict(color='Blue', width=2)))
fig.add_trace(go.Scatter(x=NO2['year'], y=NO2['NO2'], mode='lines+markers', name='NO2',line=dict(color='Red', width=2)))
fig.add_trace(go.Scatter(x=BTX['year'], y=BTX['BTX'], mode='lines+markers', name='BTX',line=dict(color='Green', width=2)))
fig.add_trace(go.Scatter(x=CO['year'], y=CO['CO'], mode='lines+markers', name='CO',line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=PM['year'], y=PM['PM2.5'], mode='lines+markers', name='PM2.5',line=dict(color='Magenta', width=2)))
fig.add_trace(go.Scatter(x=O['year'], y=O['O3'], mode='lines+markers', name='Ozone',line=dict(color='royalblue', width=2)))
fig.update_layout(title='AIR POLLUTANTS PARTICLES IN DIFFERENT YEARS', xaxis_tickfont_size=14,yaxis=dict(title='TOTAL AMOUNT IN YEARS'))
fig.show()
cities = ['Ahmedabad','Delhi','Bengaluru','Kolkata','Hyderabad']

filtered_city_day = df[df['Date'] >= '2019-01-01']
AQI = filtered_city_day[filtered_city_day.City.isin(cities)][['Date','City','AQI','AQI_Bucket']]
AQI.head()
df_Ahmedabad = df[df['City']== 'Ahmedabad']
df_Bengaluru = df[df['City']== 'Bengaluru']
df_Delhi     = df[df['City']== 'Delhi']
df_Hyderabad = df[df['City']== 'Hyderabad']
df_Kolkata   = df[df['City']== 'Kolkata']
fig,ax=plt.subplots(figsize=(20, 10))
sns.despine(fig, left=True, bottom=True)
sns.set_context("notebook", font_scale=2, rc={"lines.linewidth": 2})
sns.distplot(df_Delhi['AQI'].iloc[::30], color="y",label = 'Delhi')
sns.distplot(df_Ahmedabad['AQI'].iloc[::30], color="b",label = 'Ahmedabad')
sns.distplot(df_Hyderabad['AQI'].iloc[::30], color="black",label = 'Hyderabad')
sns.distplot(df_Bengaluru['AQI'].iloc[::30], color="g",label = 'Bengaluru')
sns.distplot(df_Kolkata['AQI'].iloc  [::30], color="r",label = 'Kolkata')
labels = [item.get_text() for item in ax.get_xticklabels()]
ax.set_xticklabels(ax.get_xticklabels(labels), rotation=30,ha="left")
plt.rcParams["xtick.labelsize"] = 15
ax.set_title('AQI DISTRIBUTIONS FROM DIFFERENT CITIES')
ax.legend(fontsize = 14);
df_Bengaluru_2019=df_Bengaluru[df_Bengaluru['Date']>='2019-01-01']
df_Ahmedabad_2019=df_Ahmedabad[df_Ahmedabad['Date']>='2019-01-01']
df_Delhi_2019=df_Delhi[df_Delhi['Date']>='2019-01-01']
df_Kolkata_2019=df_Kolkata[df_Kolkata['Date']>='2019-01-01']
df_Hyderabad_2019=df_Hyderabad[df_Hyderabad['Date']>='2019-01-01']
x = df_Ahmedabad_2019
y = df_Bengaluru_2019
z = df_Hyderabad_2019

data = [go.Scatterpolar(
  r = [x['SO2'].values[0],x['NO2'].values[0],x['CO'].values[0],x['BTX'].values[0],x['PM2.5'].values[0]],
  theta = ['SO2','NO2','CO','BTX','PM2.5'],
  fill = 'toself', opacity = 0.8,
  name = "Ahmedabad"),
        
    go.Scatterpolar(
  r = [y['SO2'].values[0],y['NO2'].values[0],y['CO'].values[0],y['BTX'].values[0],y['PM2.5'].values[0]],
  theta = ['SO2','NO2','CO','BTX','PM2.5'],
  fill = 'toself',subplot = "polar2",
    name = "Bengaluru"),
       
    go.Scatterpolar(
  r = [z['SO2'].values[0],z['NO2'].values[0],z['CO'].values[0],z['BTX'].values[0],z['PM2.5'].values[0]],
  theta = ['SO2','NO2','CO','BTX','PM2.5'],
  fill = 'toself',subplot = "polar3",
    name = "Hyderbad")]
layout = go.Layout(title = "Comparison Between Ahmedabad,Bengaluru,Hyderabad in the year 2019",
                   
                   polar = dict(radialaxis = dict(visible = True,range = [0, 120]),
                   domain = dict(x = [0, 0.27],y = [0, 1])),
                  
                   polar2 = dict(radialaxis = dict(visible = True,range = [0, 60]),
                   domain = dict(x = [0.35, 0.65],y = [0, 1])),
                  
                   polar3 = dict(radialaxis = dict(visible = True,range = [0, 70]),
                   domain = dict(x = [0.75, 1.0],y = [0, 1])),)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
df_Bengaluru_2020=df_Bengaluru[df_Bengaluru['Date']>='2020-01-01']
df_Ahmedabad_2020=df_Ahmedabad[df_Ahmedabad['Date']>='2020-01-01']
df_Delhi_2020=df_Delhi[df_Delhi['Date']>='2020-01-01']
df_Kolkata_2020=df_Kolkata[df_Kolkata['Date']>='2020-01-01']
df_Hyderabad_2020=df_Hyderabad[df_Hyderabad['Date']>='2020-01-01']
x = df_Ahmedabad_2020
y = df_Bengaluru_2020
z = df_Hyderabad_2020

data = [go.Scatterpolar(
  r = [x['SO2'].values[0],x['NO2'].values[0],x['CO'].values[0],x['BTX'].values[0],x['PM2.5'].values[0]],
  theta = ['SO2','NO2','CO','BTX','PM2.5'],
  fill = 'toself', opacity = 0.8,
  name = "Ahmedabad"),
        
    go.Scatterpolar(
  r = [y['SO2'].values[0],y['NO2'].values[0],y['CO'].values[0],y['BTX'].values[0],y['PM2.5'].values[0]],
  theta = ['SO2','NO2','CO','BTX','PM2.5'],
  fill = 'toself',subplot = "polar2",
    name = "Bengaluru"),
       
    go.Scatterpolar(
  r = [z['SO2'].values[0],z['NO2'].values[0],z['CO'].values[0],z['BTX'].values[0],z['PM2.5'].values[0]],
  theta = ['SO2','NO2','CO','BTX','PM2.5'],
  fill = 'toself',subplot = "polar3",
    name = "Hyderbad")]
layout = go.Layout(title = "Comparison Between Ahmedabad,Bengaluru,Hyderabad in the year 2020",
                   
                   polar = dict(radialaxis = dict(visible = True,range = [0, 85]),
                   domain = dict(x = [0, 0.27],y = [0, 1])),
                  
                   polar2 = dict(radialaxis = dict(visible = True,range = [0, 45]),
                   domain = dict(x = [0.35, 0.65],y = [0, 1])),
                  
                   polar3 = dict(radialaxis = dict(visible = True,range = [0, 45]),
                   domain = dict(x = [0.75, 1.0],y = [0, 1])),)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
fig=go.Figure()
fig.add_trace(go.Scatter(x=df_Delhi_2020['Date'], y=df_Delhi_2020['SO2'], mode='lines', name='SO2',line=dict(color='Blue', width=2)))
fig.add_trace(go.Scatter(x=df_Delhi_2020['Date'], y=df_Delhi_2020['NO2'], mode='lines', name='NO2',line=dict(color='Red', width=2)))
fig.add_trace(go.Scatter(x=df_Delhi_2020['Date'], y=df_Delhi_2020['BTX'], mode='lines', name='BTX',line=dict(color='Green', width=2)))
fig.add_trace(go.Scatter(x=df_Delhi_2020['Date'], y=df_Delhi_2020['CO'], mode='lines', name='CO',line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=df_Delhi_2020['Date'], y=df_Delhi_2020['PM2.5'], mode='lines', name='PM2.5',line=dict(color='Magenta', width=2)))
fig.add_trace(go.Scatter(x=df_Delhi_2020['Date'], y=df_Delhi_2020['O3'], mode='lines', name='Ozone',line=dict(color='royalblue', width=2)))
fig.update_layout(title='AIR POLLUTANTS PARTICLES ON 2020 DELHI', xaxis_tickfont_size=14,yaxis=dict(title='AIR POLLUTANTS'))
fig.show()
fig=go.Figure()
fig.add_trace(go.Scatter(x=df_Kolkata_2020['Date'], y=df_Kolkata_2020['SO2'], mode='lines', name='SO2',line=dict(color='Blue', width=2)))
fig.add_trace(go.Scatter(x=df_Kolkata_2020['Date'], y=df_Kolkata_2020['NO2'], mode='lines', name='NO2',line=dict(color='Red', width=2)))
fig.add_trace(go.Scatter(x=df_Kolkata_2020['Date'], y=df_Kolkata_2020['BTX'], mode='lines', name='BTX',line=dict(color='Green', width=2)))
fig.add_trace(go.Scatter(x=df_Kolkata_2020['Date'], y=df_Kolkata_2020['CO'], mode='lines', name='CO',line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=df_Kolkata_2020['Date'], y=df_Kolkata_2020['PM2.5'], mode='lines', name='PM2.5',line=dict(color='Magenta', width=2)))
fig.add_trace(go.Scatter(x=df_Kolkata_2020['Date'], y=df_Kolkata_2020['O3'], mode='lines', name='Ozone',line=dict(color='royalblue', width=2)))
fig.update_layout(title='AIR POLLUTANTS PARTICLES ON 2020 Kolkata', xaxis_tickfont_size=14,yaxis=dict(title='AIR POLLUTANTS'))
fig.show()
fig=go.Figure()
fig.add_trace(go.Scatter(x=df_Bengaluru_2020['Date'], y=df_Bengaluru_2020['SO2'], mode='lines', name='SO2',line=dict(color='Blue', width=2)))
fig.add_trace(go.Scatter(x=df_Bengaluru_2020['Date'], y=df_Bengaluru_2020['NO2'], mode='lines', name='NO2',line=dict(color='Red', width=2)))
fig.add_trace(go.Scatter(x=df_Bengaluru_2020['Date'], y=df_Bengaluru_2020['BTX'], mode='lines', name='BTX',line=dict(color='Green', width=2)))
fig.add_trace(go.Scatter(x=df_Bengaluru_2020['Date'], y=df_Bengaluru_2020['CO'], mode='lines', name='CO',line=dict(color='orange', width=2)))
fig.add_trace(go.Scatter(x=df_Bengaluru_2020['Date'], y=df_Bengaluru_2020['PM2.5'], mode='lines', name='PM2.5',line=dict(color='Magenta', width=2)))
fig.add_trace(go.Scatter(x=df_Bengaluru_2020['Date'], y=df_Bengaluru_2020['O3'], mode='lines', name='Ozone',line=dict(color='royalblue', width=2)))
fig.update_layout(title='AIR POLLUTANTS PARTICLES ON 2020 Bengaluru', xaxis_tickfont_size=14,yaxis=dict(title='AIR POLLUTANTS'))
fig.show()
