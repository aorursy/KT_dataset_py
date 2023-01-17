import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import folium

import os

import json



import warnings

warnings.filterwarnings('ignore')



import plotly.graph_objects as go

pd.set_option('display.max_rows',20000, 'display.max_columns',100)
us_data = pd.read_csv("../input/us-counties-covid-19-dataset/us-counties.csv")

utm_us = pd.read_csv("../input/us-utm/UTM of US States.csv")

us_data['active cases'] = us_data['cases'] - us_data['deaths']
new_us_data= us_data[us_data['date']=='2020-04-10']

us_data['date'] =pd.to_datetime(us_data.date,dayfirst=True)

us_data.info()
#Total Case in US States

carona_data = new_us_data.groupby('state')['cases','deaths','active cases'].sum().reset_index().sort_values('cases',ascending=False)

df_new = carona_data.merge(utm_us, left_on='state', right_on='state')

fig = go.Figure(data=[go.Scatter(

    x=df_new['state'][0:36],

    y=df_new['cases'][0:36],

    mode='markers',

    marker=dict(

        size=[100,90,80, 70, 60, 50, 40,35,35,35,35,35,35,35,35,35,35,30,28,28,25,25,20,15,15,15,15,10,10,10,10,10,10,10,10,10],

        color=[50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50],

        showscale=True

        )

)])

fig.update_layout(

    title='Total Case in US States',

    xaxis_title="States",

    yaxis_title="Total Cases",

)

fig.show()
#Active Case in US States

fig = go.Figure(data=[go.Scatter(

    x=df_new['state'][0:36],

    y=df_new['active cases'][0:36],

    mode='markers',

    marker=dict(

        size=[100,90,80,70,60,50,40,35,35,35,35,35,35,35,35,35,35,30,28,28,25,25,20,15,15,15,15,10,10,10,10,10,10,10,10,10],



        showscale=True

        )

)])

fig.update_layout(

    title='Active Case in US States',

    xaxis_title="States",

    yaxis_title="Active Cases",

)

fig.show()
#Daily Cases in USA Datewise

carona_data = us_data.groupby(['date'])['cases'].sum().reset_index().sort_values('cases',ascending = True)

carona_data['Daily Cases'] = carona_data['cases'].sub(carona_data['cases'].shift())

carona_data['Daily Cases'].iloc[0] = carona_data['cases'].iloc[0]

carona_data['Daily Cases'] = carona_data['Daily Cases'].astype(int)

fig = px.bar(carona_data, y='Daily Cases', x='date',hover_data =['Daily Cases'], color='Daily Cases', height=500)

fig.update_layout(

    title='Daily Cases in USA Datewise')

fig.show()
#DAILY DEATHS IN USA

carona_data = us_data.groupby(['date'])['deaths'].sum().reset_index().sort_values('deaths',ascending = True)

carona_data['DAILY DEATHS'] = carona_data['deaths'].sub(carona_data['deaths'].shift())

carona_data['DAILY DEATHS'].iloc[0] = carona_data['deaths'].iloc[0]

carona_data['DAILY DEATHS'] = carona_data['DAILY DEATHS'].astype(int)

fig = px.bar(carona_data, y='DAILY DEATHS', x='date',hover_data =['DAILY DEATHS'], color='DAILY DEATHS', height=500)

fig.update_layout(

    title='DAILY DEATHS IN USA')

fig.show()
#Corona Growth Rate(in Percentage) Comparison with Previous Day

carona_data = us_data.groupby(['date'])['cases'].sum().reset_index().sort_values('cases',ascending = True)

carona_data['Corona PD Growth Rate'] = carona_data['cases'].pct_change().mul(100).round(2)

fig = px.bar(carona_data, y='Corona PD Growth Rate', x='date',hover_data =['Corona PD Growth Rate','cases'], height=500)

fig.update_layout(

    title='Corona Growth Rate(in Percentage) Comparison with Previous Day')

fig.show()
#Corona Mortality Rate in USA

carona_data = us_data.groupby(['date'])['cases','active cases','deaths'].sum().reset_index().sort_values('date',ascending=False)

carona_data['Mortality Rate'] = ((carona_data['deaths']/carona_data['cases'])*100)

fig = go.Figure()

fig.add_trace(go.Scatter(x=carona_data['date'], y=carona_data['Mortality Rate'],

                    mode='lines+markers',name='cases',marker_color='red'))

fig.update_layout(title_text='Corona Mortality Rate in USA',plot_bgcolor='rgb(225,230,255)')

fig.show()
#Total Cases,Active Cases,Cured,Deaths from Corona Virus in USA

carona_data = us_data.groupby(['date'])['cases','active cases','deaths'].sum().reset_index().sort_values('date',ascending=False)



fig = go.Figure()

fig.add_trace(go.Scatter(x=carona_data['date'], y=carona_data['cases'],

                    mode='lines+markers',name='cases'))

fig.add_trace(go.Scatter(x=carona_data['date'], y=carona_data['active cases'], 

                mode='lines+markers',name='active cases'))

fig.add_trace(go.Scatter(x=carona_data['date'], y=carona_data['deaths'], 

                mode='lines+markers',name='deaths'))

fig.update_layout(title_text='Curve Showing Different Cases from COVID-19 in USA',plot_bgcolor='rgb(225,230,255)')

fig.show()
#USA Map with Statewise data of Total Cases,Deaths

USA_map = folium.Map(location=[37.0902,-95.7129],zoom_start=3)

fg=folium.FeatureGroup(name="my map")

fg.add_child(folium.GeoJson(data=(open('../input/usa-json/us_state00.json','r',encoding='utf-8-sig').read())))

USA_map.add_child(fg)

for lat,lan,name,deaths,cases in zip(df_new['latitude'],df_new['longitude'],df_new['state'],df_new['deaths'],df_new['cases']):

    if(deaths == 0):

        folium.Marker(location=[lat,lan],popup="<b>State  : </b>"+name+ "<br> <b>Total Cases : </b> "+str(cases)+"<br> <b>Deaths : </b> "+str(deaths)).add_to(USA_map)

    else:

        folium.Marker(location=[lat,lan],popup="<b>State  : </b>"+name+ "<br> <b>Total Cases : </b> "+str(cases)+"<br> <b>Deaths : </b> "+str(deaths),icon=folium.Icon(color="red")).add_to(USA_map)

USA_map