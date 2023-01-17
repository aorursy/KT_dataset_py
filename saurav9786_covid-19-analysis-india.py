# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import folium

from folium import plugins

import matplotlib.pyplot as plt



import math

import random

from datetime import timedelta



import warnings

warnings.filterwarnings("ignore")



#Color Palletes



cnf = '#393e46'

dth = '#ff2e63'

rec = '#21bf73'

act = '#fe9801'

new ='#25af83'



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
covid_cases_india_df= pd.read_csv("/kaggle/input/coronavirus-cases-in-india/Covid cases in India.csv")

covid_perday_df=pd.read_excel("/kaggle/input/coronavirus-cases-in-india/per_day_cases.xlsx")

covid_coordinates =pd.read_csv("/kaggle/input/coronavirus-cases-in-india/Indian Coordinates.csv")
covid_cases_india_df.head()
print(covid_cases_india_df.shape)

print(covid_perday_df.shape)

print(covid_coordinates.shape)
# Total Confirmed Cases 



confirmed = covid_cases_india_df.groupby('Name of State / UT').sum()['Total Confirmed cases'].reset_index()

confirmed
figure = px.bar(confirmed.sort_values('Total Confirmed cases', ascending= False).sort_values('Total Confirmed cases', ascending=True),x='Total Confirmed cases',y = 'Name of State / UT',title='Total Confirmed Cases',orientation='h',color_discrete_sequence=[cnf])

figure.update_layout(height=700)

figure.show()
#Total Active cases



active = covid_cases_india_df.groupby('Name of State / UT').sum()['Active'].reset_index()

active
figure = px.bar(active.sort_values('Active', ascending= False).sort_values('Active', ascending=True),x='Active',y = 'Name of State / UT',title='Total Active Cases',orientation='h',color_discrete_sequence=[act])

figure.update_layout(height=700)

figure.show()
deaths= covid_cases_india_df.groupby('Name of State / UT').sum()['Deaths'].reset_index()

deaths
figure = px.bar(deaths.sort_values('Deaths', ascending= False).sort_values('Deaths', ascending=True),x='Deaths',y = 'Name of State / UT',title='Total Death Cases',orientation='h',color_discrete_sequence=[dth])

figure.update_layout(height=700)

figure.show()
recovered= covid_cases_india_df.groupby('Name of State / UT').sum()['Cured/Discharged/Migrated'].reset_index()

recovered
figure = px.bar(recovered.sort_values('Cured/Discharged/Migrated', ascending= False).sort_values('Cured/Discharged/Migrated', ascending=True),x='Cured/Discharged/Migrated',y = 'Name of State / UT',title='Total Recovered Cases',orientation='h',color_discrete_sequence=[rec])

figure.update_layout(height=700)

figure.show()
#Confirmed Vs Recovered



fig = go.Figure(

    data=[

        go.Bar(

            name="Confirmed",

            x=covid_cases_india_df["Name of State / UT"],

            y=covid_cases_india_df["Total Confirmed cases"],

            offsetgroup=0,

        ),

        go.Bar(

            name="Recovered",

            x=covid_cases_india_df["Name of State / UT"],

            y=covid_cases_india_df["Cured/Discharged/Migrated"],

            offsetgroup=0,

            base=covid_cases_india_df["Cured/Discharged/Migrated"],

        )

    ],

    layout=go.Layout(

        title="Confirmed Vs Recovered",

        yaxis_title="Number of Cases"

    )

)

fig.show()
#Confirmed Vs Deaths



fig = go.Figure(

    data=[

        go.Bar(

            name="Confirmed",

            x=covid_cases_india_df["Name of State / UT"],

            y=covid_cases_india_df["Total Confirmed cases"],

            offsetgroup=0,

        ),

        go.Bar(

            name="Deaths",

            x=covid_cases_india_df["Name of State / UT"],

            y=covid_cases_india_df["Deaths"],

            offsetgroup=0,

            base=covid_cases_india_df["Deaths"],

        )

    ],

    layout=go.Layout(

        title="Confirmed Vs Deaths",

        yaxis_title="Number of Cases"

    )

)

fig.show()
covid_coordinates.head()
covid_coordinates.drop('Unnamed: 3', axis=1,inplace=True)
covid_coordinates.head()
center_lat = covid_coordinates.mean().Latitude

center_long = covid_coordinates.mean().Longitude
laMap = folium.Map(location=[center_lat, center_long], tiles='Stamen Toner', zoom_start=5)

for i,row in covid_coordinates.iterrows():

    folium.CircleMarker((row.Latitude,row.Longitude), radius=3, weight=2, color='red', fill_color='red', fill_opacity=.5).add_to(laMap)

laMap
laMap = folium.Map(location=[center_lat, center_long], tiles='Stamen Toner', zoom_start=5)

for i,row in covid_coordinates.iterrows():

    folium.CircleMarker((row.Latitude,row.Longitude), radius=3, weight=2, color='red', fill_color='red', fill_opacity=.5).add_to(laMap)

laMap.add_children(plugins.HeatMap(data=covid_coordinates[['Latitude', 'Longitude']].to_numpy (), radius=25, blur=10))

laMap
covid_perday_df.head()
fig =go.Figure()

fig.add_trace(go.Scatter(x=covid_perday_df['Date'],y =covid_perday_df['Total Cases'], mode='lines+markers',name= 'Total Cases',line=dict(color="Black",width=2)))

fig.add_trace(go.Scatter(x=covid_perday_df['Date'],y =covid_perday_df['New Cases'], mode='lines+markers',name= 'New Cases',line=dict(color="Orange",width=2)))

fig.update_layout(title='India Covid-19 cases',xaxis_tickfont_size =14,yaxis=dict(title="No of cases"))

fig.show()
temp = covid_perday_df.groupby('Date')['Total Cases','New Cases'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop= True)

tm = temp.melt(id_vars='Date',value_vars=['Total Cases','New Cases'])

fig = px.treemap(tm,path=["variable"],values='value',height=250,width = 800,color_discrete_sequence=[cnf,new])

fig.data[0].textinfo ='label+text+value'

fig.show()
temp = covid_perday_df.groupby('Date')['Total Cases','New Cases'].sum().reset_index()

temp = temp.melt(id_vars= "Date",value_vars= ['Total Cases','New Cases'],var_name='Case',value_name="Count")

fig = px.area(temp,x="Date",y="Count",color="Case",height=400,title="Cases Over time",color_discrete_sequence=[cnf,new])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
fig_t = px.bar(covid_perday_df,x='Date',y = 'Total Cases',color_discrete_sequence=[cnf])

fig_n = px.bar(covid_perday_df,x='Date',y = 'New Cases',color_discrete_sequence=[new])

fig = make_subplots(rows=1,cols=2,shared_xaxes=False,horizontal_spacing=0.1,

                   subplot_titles=('Total Cases','New Cases'))

fig.add_trace(fig_t['data'][0],row=1,col=1)

fig.add_trace(fig_n['data'][0],row=1,col=2)



fig.update_layout(height=400)

fig.show()