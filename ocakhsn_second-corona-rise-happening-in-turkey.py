import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import datetime

from datetime import date, timedelta



%matplotlib inline
df = pd.read_csv('/kaggle/input/corona-virus-test-numbers-in-turkey/turkey_covid19_all.csv')
df.head()
df.isnull().sum()
df['Longitude'].value_counts()
import folium



m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()],  zoom_start=6)



folium.CircleMarker(

    location=[df['Latitude'].mean(), df['Longitude'].mean()],

    radius=50,

    tooltip='<li><bold>Country : Turkey'+

                    '<li><bold>Confirmed : '+str(df['Confirmed'].max())+

                    '<li><bold>Deaths : '+str(df['Deaths'].max())+

                    '<li><bold>Recovered : '+str(df['Recovered'].max()),

    color='#3186cc',

    fill=True,

    fill_color='#3186cc'

).add_to(m)



m
df['Active'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
values=[df['Active'].iloc[-1], df['Recovered'].iloc[-1], df['Deaths'].iloc[-1]]

colors = ['gold', 'mediumturquoise', 'darkorange']



fig = go.Figure(data=[go.Pie(labels=['Active','Recovered','Deaths'],

                             values=values)])

fig.update_traces(hoverinfo='label+percent', textinfo='label+value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text='Distribution of the Cases in Turkey', template='plotly_dark')

fig.show()

values=[df['Confirmed'].iloc[-1] -df['Confirmed'].iloc[-2], df['Recovered'].iloc[-1] - df['Recovered'].iloc[-2], df['Deaths'].iloc[-1] -df['Deaths'].iloc[-2]]

colors = ['gold', 'mediumturquoise', 'darkorange']



fig = go.Figure(data=[go.Pie(labels=['Confirmed','Recovered','Deaths'],

                             values=values)])

fig.update_traces(hoverinfo='label+percent', textinfo='label+value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text='Daily '+ df['Date'].iloc[-1] +' Statistics', template='plotly_dark')

fig.show()

dates = df.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active', 'Tests'].sum()

dates.reset_index(inplace=True)

print(dates.shape)
dates.columns
fig = px.bar(dates,x='Date',y='Confirmed',title='Total Confirmed Cases',color_discrete_sequence=['#5DADE2'])

fig.update_layout(template='plotly_dark')

fig.show()
dates['New_Cases'] = dates['Confirmed'] - dates['Confirmed'].shift(1)

fig = px.bar(dates,x='Date',y='New_Cases',title='Case Number in Each Day',color_discrete_sequence=['#1E8449'])

fig.update_layout(template='plotly_dark')

fig.show()
fig = px.bar(dates,x='Date',y='Deaths',title='Total Death Cases',color_discrete_sequence=['#7D3C98'])

fig.update_layout(template='plotly_dark')

fig.show()
dates['New_Deaths'] = dates['Deaths'] - dates['Deaths'].shift(1)

fig = px.bar(dates,x='Date',y='New_Deaths',title='Death Numbers in Each Day',color_discrete_sequence=['#E67E22'])

fig.update_layout(template='ggplot2')

fig.show()
fig = px.bar(dates,x='Date',y='Tests',title='Test Numbers in Each Day',color_discrete_sequence=['#5D6D7E'])

fig.update_layout(template='presentation')

fig.show()
fig = px.bar(dates,x='Date',y='Recovered',title='Total Recovered Cases',color_discrete_sequence=['#5199FF'])

fig.update_layout(template='plotly_dark')

fig.show()
dates['Recovered_Day'] = dates['Recovered'] - dates['Recovered'].shift(1)

fig = px.bar(dates,x='Date',y='Recovered_Day',title='Recovered Numbers in Each Day',color_discrete_sequence=['#C0392B'])

fig.update_layout(template='ggplot2')

fig.show()
fig = go.Figure(data=[

go.Bar(name='Tested', y=dates['Date'], x=dates['Tests'], orientation='h'),

go.Bar(name='Positive', y=dates['Date'], x=dates['New_Cases'], orientation='h')])

fig.update_layout(barmode='stack',width=900, height=600)

fig.update_layout(title_text='Tested People vs Positive Cases')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Confirmed'],

                    mode='lines+markers',marker_color='blue',name='Total Cases'))



fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Recovered'],

                    mode='lines+markers',marker_color='green',name='Recovered Cases'))



fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Deaths'],

                    mode='lines+markers',marker_color='red',name='Deaths Cases'))



fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Active'],

                    mode='lines+markers',marker_color='#17202A',name='Active Cases'))



fig.update_layout(title_text='Corona Virus Statistics in Turkey', template='ggplot2')

fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x=dates['Date'], y=dates['New_Cases'],

                    mode='lines+markers',marker_color='blue',name='Daily New Cases'))



fig.add_trace(go.Scatter(x=dates['Date'], y=dates['New_Deaths'],

                    mode='lines+markers',marker_color='red',name='Daily Deaths'))



fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Recovered_Day'],

                    mode='lines+markers',marker_color='green',name='Daily Recovered'))

fig.update_layout(title_text='Daily Statistics', template='ggplot2')



fig.show()


dates['Mortality Rate'] = (dates['Deaths'] / dates['Confirmed']) * 100

dates['Recovery Rate'] = (dates['Recovered'] / dates['Confirmed']) * 100

dates['Spread Rate'] = (dates['New_Cases'] / dates['Confirmed'].shift(1)) * 100

dates['Positive Rate'] = (dates['New_Cases'] / dates['Tests'])  * 100

fig = go.Figure()

fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Spread Rate'],

                    mode='lines+markers',marker_color='#1C32AE',name='Spread Rate'))

fig.update_layout(title_text='Spread Rate in Turkey', template='ggplot2')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Mortality Rate'],

                    mode='lines+markers',marker_color='green',name='Mortality Rate'))

fig.update_layout(title_text='Mortality Rate in Turkey', template='ggplot2')

fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Recovery Rate'],

                    mode='lines+markers',marker_color='#B96BAA',name='Recovery Rate'))

fig.update_layout(title_text='Recovery Rate in Turkey', template='ggplot2')

fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Positive Rate'],

                    mode='lines+markers',marker_color='#FF005C',name='Positive Test Rate'))

fig.update_layout(title_text='Positive Rate in Turkey', template='ggplot2')

fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Spread Rate'],

                    mode='lines+markers',marker_color='pink',name='Spread Rate'))



fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Mortality Rate'],

                    mode='lines+markers',marker_color='red',name='Mortality Rate'))



fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Recovery Rate'],

                    mode='lines+markers',marker_color='green',name='Recovery Rate'))



fig.add_trace(go.Scatter(x=dates['Date'], y=dates['Positive Rate'],

                    mode='lines+markers',marker_color='blue',name='Positive Rate'))

fig.update_layout(title_text='Rate Statistics', template='ggplot2')



fig.show()
temp = dates.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Confirmed'],

                 var_name='Case', value_name='Count')



fig = px.area(temp, x="Date", y="Count", color='Case',

             title='Cases over time in Turkey', color_discrete_sequence = ['#A93226','#17202A','#B96BAA'])

fig.update_layout(template='ggplot2')

fig.show()
dates.head()
from fbprophet import Prophet

sub_data = dates[['Date', 'New_Cases']]



sub_data['New_Cases'] = sub_data['New_Cases'].fillna(0)

sub_data.rename(columns={'Date': 'ds', 'New_Cases': 'y'}, inplace=True)

sub_data.head()
m = Prophet(weekly_seasonality=False)

m.fit(sub_data)

future = m.make_future_dataframe(periods=30)

future.head()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
from fbprophet.plot import plot_plotly, plot_components_plotly



plot_plotly(m, forecast)
fig2 = m.plot_components(forecast)