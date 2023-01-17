# Import necessary libraries

import gpxpy.geo #Get the haversine distance

import math

import pickle

import os

import pandas as pd

import folium 

import datetime #Convert to unix time

import time #Convert to unix time

import numpy as np#Do aritmetic operations on arrays

# matplotlib: used to plot graphs

import matplotlib

# matplotlib.use('nbagg') : matplotlib uses this protocall which makes plots more user intractive like zoom in and zoom out

matplotlib.use('nbagg')

import matplotlib.pylab as plt

import seaborn as sns#Plots

from matplotlib import rcParams#Size of plots 

import plotly as py

import cufflinks

from tqdm import tqdm_notebook as tqdm
# Reading Data

covid_master=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covid_open=pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

covid_confirmed=pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

covid_death= pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

covid_master.head(3)

#data[data['ObservationDate']=='03/04/2020']
# data columns

#covid_master=covid_master.drop(columns=['SNo'])

covid_master.columns
covid_master.isna().sum()
# We will replace Null states to a value 'NoState'

covid_master=covid_master.fillna('NoState')

covid_master.isna().sum()
# changing the data type

num_cols=['Confirmed', 'Deaths', 'Recovered']

for col in num_cols:

    temp=[int(i) for i in covid_master[col]]

    covid_master[col]=temp
covid_master.groupby(['Country/Region','Confirmed']).sum()
# Consolidating unique affected regions till date

unique_regions=[country for country in list(set(covid_master['Country/Region']))]
# Creating list of all regions of all counntries

train=covid_master

unique_regions=train['Country/Region'].unique()

states_per_regions=[]

for reg in tqdm(unique_regions):

    states_per_regions.append(train[train['Country/Region']==reg]['Province/State'].unique()) 

print('No of unique regions:',len(unique_regions))  
# Total Confirmed cases per conutry

total_confirmed=[]

for i in range(len(unique_regions)):

    count=0

    covid_temp=covid_master[covid_master['Country/Region']==unique_regions[i]]

    for state in states_per_regions[i]:

        #print(state)

        count+=covid_temp[covid_temp['Province/State']==state].sort_values('ObservationDate').iloc[-1]['Confirmed']

    total_confirmed.append(count)
# Total Deaths cases per conutry

total_deaths=[]

for i in range(len(unique_regions)):

    count=0

    covid_temp=covid_master[covid_master['Country/Region']==unique_regions[i]]

    for state in states_per_regions[i]:

        #print(state)

        count+=covid_temp[covid_temp['Province/State']==state].sort_values('ObservationDate').iloc[-1]['Deaths']

    total_deaths.append(count)
# Total Recovered cases per conutry

total_recovered=[]

for i in range(len(unique_regions)):

    count=0

    covid_temp=covid_master[covid_master['Country/Region']==unique_regions[i]]

    for state in states_per_regions[i]:

        #print(state)

        count+=covid_temp[covid_temp['Province/State']==state].sort_values('ObservationDate').iloc[-1]['Recovered']

    total_recovered.append(count)
# We will ignore the data of 'Diamond pricess Crusie' for countrywise analysis as the data is unconfirmed

covid_countrywise=pd.DataFrame(columns=['country','confirmed','deaths','recovered'],index=None)

unique_regions[53]='unconfirmed/Diamond princes Cruise'

total_confirmed[53]=0

total_recovered[53]=0

total_deaths[53]=0

covid_countrywise.country=unique_regions

covid_countrywise.confirmed=total_confirmed

covid_countrywise.deaths=total_deaths

covid_countrywise.recovered=total_recovered

covid_countrywise.to_csv('covid_countrywise.csv')

covid_countrywise.head()
# This function calculates number of confirmed/death/recovered cases before nth day

# To calculate the number of cases on last day pass '-1', to  get data of 2 days ago pass '-2'

# The function will return a list of case type by uniique countries

def num_cases_n_days_before(n,case_type,unique_regions):

    """

    case_type ='Confirmed/Deaths/Recovered'

    n= -1,-2 etc.

    uniue_regions= list of all unique regions/Countries

    """

    total_cases=[]

    for i in range(len(unique_regions)):

        count=0

        covid_temp=covid_master[covid_master['Country/Region']==unique_regions[i]]

        for state in states_per_regions[i]:

            if(len(covid_temp[covid_temp['Province/State']==state])>abs(n)):

                count+=covid_temp[covid_temp['Province/State']==state].sort_values('ObservationDate').iloc[-2]['Confirmed']

        total_cases.append(count)

    return total_cases
def get_countrywise_spike_score(unique_regions):

    num_latest_confirmed_cases_=num_cases_n_days_before(-1,'Confirmed',unique_regions)

    num_confirmed_cases_2_days_ago_=num_cases_n_days_before(-3,'Confirmed',unique_regions)

    num_confirmed_cases_6_days_ago_=num_cases_n_days_before(-7,'Confirmed',unique_regions)

    #print(num_latest_confirmed_cases)

    num_latest_confirmed_cases=[i+1 for i in num_latest_confirmed_cases_]

    num_confirmed_cases_2_days_ago=[i+1 for i in  num_confirmed_cases_2_days_ago_]

    num_confirmed_cases_6_days_ago=[i+1 for i in num_confirmed_cases_6_days_ago_ ]

    spike_scores=[]

    for i in range(len(num_latest_confirmed_cases)):

        spike_1=((num_latest_confirmed_cases[i]-num_confirmed_cases_2_days_ago[i])/num_confirmed_cases_2_days_ago[i])*100

        spike_2=((num_confirmed_cases_2_days_ago[i]-num_confirmed_cases_6_days_ago[i])/num_confirmed_cases_6_days_ago[i])*100

        spike_scores.append(spike_1+spike_2)

    covid_countrywise_spike_score=pd.DataFrame(columns=['Country','Spike_score','num_latest_confirmed','confirmed_2_days_ago','confirmed_6_days_ago'],index=None)

    covid_countrywise_spike_score.Country=unique_regions

    covid_countrywise_spike_score.Spike_score=spike_scores

    covid_countrywise_spike_score.num_latest_confirmed=num_latest_confirmed_cases_

    covid_countrywise_spike_score.confirmed_2_days_ago=num_confirmed_cases_2_days_ago_

    covid_countrywise_spike_score.confirmed_6_days_ago=num_confirmed_cases_6_days_ago_

    return covid_countrywise_spike_score
covid_countrywise_spike_score=get_countrywise_spike_score(unique_regions)

#covid_countrywise_spike_score.sort_values('Spike_score',ascending=False)[0:10])
#covid_countrywise.iloc[53]['Country']='unconfirmed/Diamond princes Cruise'

covid_countrywise.sort_values('confirmed',ascending=False).head(6)
covid_countrywise_top=covid_countrywise.sort_values('confirmed',ascending=False).head(6)
import plotly.express as px

data=covid_countrywise.sort_values('confirmed',ascending=False)[0:7]



fig = px.bar(data, x='country', y='confirmed',

             hover_data=['country','confirmed'], color='confirmed',

             labels={'pop':'Confirmed Cases'}, height=400,title='Seven worstly hit countries')

fig.update_layout(template='plotly_dark')

fig.show()
# Counting total confirmed, deaths and recovered cases for rest of the world

data=covid_countrywise.sort_values('confirmed',ascending=False)

row_confirmed=0

row_deaths=0

row_recovered=0

for i in range(1,len(covid_countrywise)):

    row_confirmed+=data.iloc[i]['confirmed']

    row_deaths+=data.iloc[i]['deaths']

    row_recovered+=data.iloc[i]['recovered']
Us=[data.iloc[0]['confirmed'],data.iloc[0]['recovered'],data.iloc[0]['deaths']]

rest_of_the_world=[row_confirmed,row_recovered,row_deaths]
#https://plot.ly/python/bar-charts/

import plotly.graph_objects as go

data=covid_countrywise.sort_values('confirmed',ascending=False)[0:10]

country=data['country']



fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Bar(x=['US','Rest of the world'],

                y=[Us[0],rest_of_the_world[0]],

                name='Confirmed',

                marker_color='rgb(102, 102, 255)'

                ))

fig.add_trace(go.Bar(x=['US','Rest of the world'],

                y=[Us[1],rest_of_the_world[1]],

                name='Recovered',

                marker_color='rgb(0,255,153)'

                ))

fig.add_trace(go.Bar(x=['US','Rest of the world'],

                y=[Us[2],rest_of_the_world[2]],

                name='Deaths',

                marker_color='rgb(255, 102, 102)'

                ))

fig.update_layout(

    title='Confirmed/Recovered/Deaths in United States and Rest of the World',

    xaxis_tickfont_size=15,

    yaxis=dict(

        title='count',

        titlefont_size=12,

        tickfont_size=15,

       

    ),

    legend=dict(

        x=1,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.10, # gap between bars of adjacent location coordinates.

    bargroupgap=0.2 # gap between bars of the same location coordinate.

)

fig.show()
print('The death rate in US is:',str((Us[2]/Us[0])*100)+' %')

print('The rate of people already recovered in US is(till date):',str((Us[1]/Us[0])*100)+' %')

print('The death rate in rest of the world is:',str((rest_of_the_world[2]/rest_of_the_world[0])*100)+' %')

print('The rate of people already recovered in rest of the world is(till date):',str((rest_of_the_world[1]/rest_of_the_world[0])*100)+' %')

print('The current overall death rate is:',str(((Us[2]+rest_of_the_world[2])/(Us[0]+rest_of_the_world[0]))*100)+' %')
fig.update_layout(barmode='relative', title_text='United States vs Rest of the World Relative Stats',bargap=0.2)

fig.show()
fig = px.pie(data, values='confirmed', names='country', title='Distribution of confirmed cases globally')

fig.show()
#https://plot.ly/python/pie-charts/

from plotly.subplots import make_subplots

data=covid_countrywise.sort_values('confirmed',ascending=False)[0:5]

labels=list(data['country'])

# Create subplots: use 'domain' type for Pie subplot

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=list(data['confirmed']), name='country'),

              1, 1)

fig.add_trace(go.Pie(labels=labels, values=list(data['deaths']), name='country'),

              1, 2)



# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Distribution of confirmed and death cases across worstly hit countries",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='confirm', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='Deaths', x=0.82, y=0.5, font_size=20, showarrow=False)])

fig.show()
data=covid_countrywise.sort_values('confirmed',ascending=False)

death_confirm_ratio=((data.deaths)/(data.confirmed))*100

data['death_rates']=death_confirm_ratio

data.sort_values('death_rates',ascending=False).head(5)
fig = px.bar(data.sort_values('death_rates',ascending=False).head(10), x='country', y='death_rates',

             hover_data=['country','death_rates'], color='country',

             labels={'pop':'Confirmed Cases'}, height=400,title='Countries with worst death rates')

fig.update_layout(template='plotly_dark')

fig.show()
data=covid_countrywise.sort_values('confirmed',ascending=False)

recovered_confirm_ratio=((data.recovered)/(data.confirmed))*100

data['recovery_rates']=recovered_confirm_ratio

data.sort_values('recovery_rates',ascending=False).head(5)
fig = px.bar(data.sort_values('recovery_rates',ascending=False).head(20), x='country', y='recovery_rates',

             hover_data=['country','recovery_rates'], color='country',

             labels={'pop':'Confirmed Cases'}, height=400,title='Countries with best recovery rates')

fig.update_layout(template='plotly_dark')

fig.show()
# Here we will calculate where the number of confirmed cases are increasing at an alarming rate

# Spike Score = percentage increase in cases from n-6 days to n-3 days + percentage increase in cases from n-3 days to n-1 days

covid_countrywise_spike_score=get_countrywise_spike_score(unique_regions)

covid_countrywise_spike_score.sort_values('Spike_score',ascending=False).head()
fig = px.bar(covid_countrywise_spike_score.sort_values('Spike_score',ascending=False).head(20), x='Country', y='Spike_score',

             hover_data=['Country','Spike_score'], color='Country',

             labels={'pop':'Confirmed Cases'}, height=400,title='Countries where spreading rate is alarming for past week')

fig.update_layout(template='plotly_dark')

fig.show()
covid_timeseries = covid_master.groupby('ObservationDate')['Confirmed', 'Deaths', 'Recovered'].sum()

covid_timeseries=covid_timeseries.reset_index().sort_values('ObservationDate')

covid_timeseries.head()
#https://plot.ly/python

x=[i.split('/')[0]+'/'+i.split('/')[1] for i in covid_timeseries.ObservationDate ]

fig = go.Figure()

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=x, 

                         y=covid_timeseries['Confirmed'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='rgb(102, 102, 255)', width=2)))

fig.add_trace(go.Scatter(x=x, 

                         y=covid_timeseries['Deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='rgb(255, 102, 102)', width=2)))

fig.add_trace(go.Scatter(x=x, 

                         y=covid_timeseries['Recovered'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='rgb(0,255,153)', width=2)))

fig.update_layout(

    title = 'Spread of COVID-19 over time',

    xaxis_tickformat = '%d %B (%a)<br>%Y'

)

fig.show()
covid_timeseries = covid_master.groupby(['ObservationDate','Country/Region'])['Confirmed', 'Deaths', 'Recovered'].sum()

covid_timeseries=covid_timeseries.reset_index().sort_values('ObservationDate')

data= covid_timeseries[covid_timeseries['Country/Region']!='Mainland China']

data.head()
fig=go.Figure()

x=[i.split('/')[1]+'/'+i.split('/')[0] for i in data.ObservationDate ]

data['ObservationDate']=x

fig.update_layout(template='plotly_dark')

fig = px.line(data, x="ObservationDate", y="Confirmed", color="Country/Region",

              line_group="Country/Region", hover_name="Country/Region")

fig.update_layout(template='plotly_dark',title_text='spreading of COVID-19 outside China')

fig.show()
covid_timeseries = covid_master.groupby('ObservationDate')['Confirmed', 'Deaths', 'Recovered'].sum()

covid_timeseries=covid_timeseries.reset_index().sort_values('ObservationDate')

covid_confirmed=covid_timeseries.Confirmed

covid_death=covid_timeseries.Deaths

covid_recovered=covid_timeseries.Recovered

Newly_reported=[covid_confirmed[0]]

New_deaths=[covid_death[0]]

New_recovered=[covid_recovered[0]]

for i in range(1,len(covid_confirmed)):

    Newly_reported.append(covid_confirmed[i]-covid_confirmed[i-1])

    New_deaths.append(covid_death[i]-covid_death[i-1])

    New_recovered.append(covid_recovered[i]-covid_recovered[i-1])

covid_timeseries['Newly Confirmed']=Newly_reported 

covid_timeseries['New Death']=New_deaths

covid_timeseries['New Recovered']=New_recovered

covid_timeseries.head()
fig=go.Figure()

x=[i.split('/')[0]+'/'+i.split('/')[1] for i in covid_timeseries.ObservationDate ]

fig.add_trace(go.Scatter(x=x, 

                         y=covid_timeseries['Newly Confirmed'],

                         mode='lines',

                         name='New Confirmed Incident',

                         line=dict(color='rgb(102, 102, 255)', width=2)))

fig.add_trace(go.Scatter(x=x,y=covid_timeseries['New Death'],name='New Death Incident',

                        mode='lines',line=dict(color='rgb(255, 102, 102)', width=2)))



fig.add_trace(go.Scatter(x=x,y=covid_timeseries['New Recovered'],name='New Recovery Incident',

             mode='lines',line=dict(color='rgb(0,255,153)', width=2)))

fig.update_layout(

    title = 'New Incident Reported/Recovered/Death per Day',

    xaxis_tickformat = '%d %B (%a)<br>%Y',template='plotly_dark'

)

fig.show()
covid_country=[con.lower() for con in covid_countrywise.country]

covid_country[23]='china'

covid_countrywise.country=covid_country
# Merging the COVID-19 data with world co-ordinate data to get the geo code data

coordinates=pd.read_csv('../input/world-coordinates/world_coordinates.csv')

coordinates=coordinates.rename(columns={'Country':'country'})

cords_country=[con.lower() for con in coordinates.country]

coordinates.country=cords_country

world_data=pd.merge(covid_countrywise,coordinates,on='country')
# We will create a world map with circles in affected regions

# We will choose the circle radius based on the confirmed ratio

total_confirmed=sum(i for i in covid_countrywise['confirmed'])

total_confirmed_countrywise=world_data.confirmed

percentage_confirmed_per_country=(total_confirmed_countrywise/total_confirmed)*100

for i in range(len(percentage_confirmed_per_country)):

    if(percentage_confirmed_per_country[i]<5):

        percentage_confirmed_per_country[i]=5

    elif(percentage_confirmed_per_country[i]>5 and percentage_confirmed_per_country[i]<25):

        percentage_confirmed_per_country[i]=10

    elif(percentage_confirmed_per_country[i]>25 and percentage_confirmed_per_country[i]<50):

        percentage_confirmed_per_country[i]=15

    else:

         percentage_confirmed_per_country[i]=20

world_data['radius']=percentage_confirmed_per_country
# create map and display it

# How to create map using Folium

#https://python-visualization.github.io/folium/modules.html

# credit: https://www.kaggle.com/parulpandey/wuhan-coronavirus-a-geographical-analysis/data

world_map = folium.Map(location=[30, 0], zoom_start=1.5,tiles='Stamen Toner')

for lat, lon, value, name,confirm_ratio in zip(world_data['latitude'], world_data['longitude'], world_data['confirmed'], world_data['country'],world_data['radius']):

    folium.CircleMarker([lat, lon],

                        radius=confirm_ratio,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),

                        color='rgb(102, 102, 255)',

                        

                        fill_color='rgb(102, 102, 255)',

                        fill_opacity=0.7 ).add_to(world_map)
world_map
#https://plot.ly/python/mixed-subplots/

# Initialize figure with subplots

fig = make_subplots(

    rows=2, cols=2,

    subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4"),

    column_widths=[0.6, 0.4],

    row_heights=[0.4, 0.6],

    specs=[[{"type": "scattergeo", "rowspan": 2}, {"type": "bar"}],

           [            None                    , {"type": "bar"}]])



# Add scattergeo globe map of volcano locations

fig.add_trace(

    go.Scattergeo(lat=world_data["latitude"],

                  lon=world_data["longitude"],

                  mode="markers",

                  hoverinfo="text",

                  text=world_data['country'],

                  showlegend=True,

                  name='Effected regions',

                  marker=dict(color="crimson", size=4, opacity=0.8)),

    row=1, col=1

)





x=[i.split('/')[0]+'/'+i.split('/')[1] for i in covid_timeseries.ObservationDate ]

fig.update_layout(template='plotly_dark')

fig.add_trace(go.Scatter(x=x, 

                         y=covid_timeseries['Confirmed'],

                         mode='lines+markers',

                         name='Confirmed',

                         line=dict(color='rgb(102, 102, 255)', width=2)),1,2)

fig.add_trace(go.Scatter(x=x, 

                         y=covid_timeseries['Deaths'],

                         mode='lines+markers',

                         name='Deaths',

                         line=dict(color='rgb(255, 102, 102)', width=2)),1,2)

fig.add_trace(go.Scatter(x=x, 

                         y=covid_timeseries['Recovered'],

                         mode='lines+markers',

                         name='Recovered',

                         line=dict(color='rgb(0,255,153)', width=2)),1,2)





fig.add_trace(go.Scatter(x=x, 

                         y=covid_timeseries['Newly Confirmed'],

                         mode='lines',

                         name='New Confirmed Incident',

                         line=dict(color='rgb(102, 102, 255)', width=2)),2,2)

fig.add_trace(go.Scatter(x=x,y=covid_timeseries['New Death'],name='New Death Incident',

                        mode='lines',line=dict(color='rgb(255, 102, 102)', width=2)),2,2)



fig.add_trace(go.Scatter(x=x,y=covid_timeseries['New Recovered'],name='New Recovery Incident',

             mode='lines',line=dict(color='rgb(0,255,153)', width=2)),2,2)



# Update geo subplot properties

fig.update_geos(

    projection_type="orthographic",

    landcolor="white",

    oceancolor="MidnightBlue",

    showocean=True,

    lakecolor="LightBlue"

)





# Rotate x-axis labels

fig.update_xaxes(tickangle=45)



# Set theme, margin, and annotation in layout

fig.update_layout(

    template="plotly_dark",

     title_text="COVID-19 World Wide Spread Quick Dashboard",

    margin=dict(r=10, t=25, b=40, l=60),

    annotations=[

        dict(

           

            text="Source:JHU",

            showarrow=False,

            xref="paper",

            yref="paper",

            x=0,

            y=0)

    ]

)



fig.show()

world_map_deaths = folium.Map(location=[30, 0], zoom_start=1.5,tiles='Stamen Toner')

world_data_deaths=world_data[world_data['deaths']>0]

death_confirm_ratio=((world_data.deaths)/(world_data.confirmed))*100

world_data_deaths['death_rates']=death_confirm_ratio

for lat, lon, value, name,rad in zip(world_data_deaths['latitude'], world_data_deaths['longitude'], world_data_deaths['deaths'], world_data_deaths['country'],world_data_deaths['death_rates']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Death Cases</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(world_map_deaths)
world_map_deaths
world_map_recovered = folium.Map(location=[30, 0], zoom_start=1.5,tiles='Stamen Toner')

world_data_totaly_recovered=world_data[world_data['confirmed']==world_data['recovered']]

for lat, lon, value, name in zip(world_data_totaly_recovered['latitude'], world_data_totaly_recovered['longitude'], world_data_totaly_recovered['recovered'], world_data_totaly_recovered['country']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Recovered</strong>: ' + str(value) + '<br>'),

                        color='green',

                        

                        fill_color='green',

                        fill_opacity=0.7 ).add_to(world_map_recovered)
world_map_recovered
covid_master.head()
# Run this cell to get the summary of latest status report 

from datetime import date

covid_countrywise_top

status_date=covid_timeseries.iloc[-1]['ObservationDate']

total_confirmed=sum(i for i in covid_countrywise['confirmed'])

total_recovered=sum(i for i in covid_countrywise['recovered'])

total_deaths=sum(i for i in covid_countrywise['deaths'])

print('------Status Report------')

print('Last Updated:',status_date)

print('Total Confirmed:',total_confirmed)

print('Total recovered:',total_recovered)

print('Total Deaths:',total_deaths)

print('Global Death Rate:'+ str(round((total_deaths/total_confirmed)*100,2))+' %')

print('No of Confirmed cases added on last day:',(covid_timeseries.iloc[-1]['Newly Confirmed']))

print('No of Death cases added on last day:',(covid_timeseries.iloc[-1]['New Death']))

print('No of Recovered cases added on last day:',(covid_timeseries.iloc[-1]['New Recovered']))
