#Loading libraries

from plotly.offline import init_notebook_mode, iplot

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from wordcloud import WordCloud

from textblob import TextBlob 

from plotly import tools

import seaborn as sns

import pandas as pd

import string, os, random

import calendar

from PIL import Image 

import numpy as np



import folium 

from folium import plugins 





init_notebook_mode(connected=True)

punc = string.punctuation

from datetime import datetime

#loading data

df = pd.read_csv("../input/us-gun-violence/gun-violence-data_01-2013_03-2018.csv")
# Adding more specific time features

df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].dt.year # a new column for year

df['month'] = df['date'].dt.month # a new column for month

df['dayinmonth'] = df['date'].dt.day # a new column for the day of the month

df['dayinweek'] = df['date'].dt.weekday_name # a new column for the day of the week



# A variable for the total number of victims for each incident = # of killed + # of injured

df['victims'] = df['n_killed'] + df['n_injured']



df.shape
df.isnull().sum()
# First, let's find out what is the most commonly used weapon in shooting incidents. Is it a automatic rifle, or a handgun?





df['gun_type_spc'] = df['gun_type'].fillna('0:Unknown') # fill all the missing values with 0:Unknown

g = df.groupby(['gun_type_spc']).agg({'n_killed': 'sum', 'n_injured' : 'sum', 'state' : 'count'}).reset_index().rename(columns={'state':'count'})

#group data in gun_type_spc, sum # of deaths, injuries by catogires in gun_type_spc,  

#count the incidents by catogies in gun_type_parsed

#reset multi-level index

#rename state to count





results={}

for i, row in g.iterrows():

    words = row['gun_type_spc'].split("||")

    for word in words:

        if "Unknown" in word:

            continue

        word = word.replace("::",":").replace("|1","")

        guntype = word.split(":")[1]

        if guntype not in results:

            results[guntype] = {'killed' : 0, 'injured' : 0, 'used' : 0}

        results[guntype]['killed'] += row['n_killed']

        results[guntype]['injured'] += row['n_injured']

        results[guntype]['used'] += row['count']



results
gun_names = list(results.keys())



used=[]

for each in list(results.values()):

    used.append(each['used'])

killed=[]

for each in list(results.values()):

    killed.append(each['killed'])

injured=[]

for each in list(results.values()):

    injured.append(each['injured'])    

#Danger measures the average number of victims per use of this type of gun

danger=[]

for i, x in enumerate(used):

    danger.append((killed[i] + injured[i]) / x)

#Plotting a bar graph

trace1 = go.Bar(x=gun_names,y=used,orientation = 'v',

    marker = dict(color = 'black', 

        line = dict(color = 'white', width = 1) ))

data = [trace1]

layout = dict(height=650, title='Types of Guns Used vs Numbers of Usage', legend=dict(orientation="v"));

fig = go.Figure(data=data, layout=layout)

iplot(fig)
a = df[df['year'].isin(['2013','2014','2015','2016','2017'])]

mc = a.groupby(['year','month']).agg({'month':'count'}).rename(columns={'month':'month_count'}).reset_index()

avg_mc = mc.groupby(['month']).agg({'month_count':'mean'})
x=list(avg_mc.index)

y=list(avg_mc.month_count)
l={}

for m,n in zip(x,y):

    l[m]=n

months=[]

mon_mean=[]

for month in l:

    months.append(calendar.month_abbr[int(month)])

    mon_mean.append(l[month])



trace = go.Bar(x=months,y=mon_mean,marker=dict(color='red',line=dict(color='red',width=1)))

data = [trace]

layout = dict(height=420,title='Average Shooting Incidents Per Month')

fig = go.Figure(data=data,layout=layout)

iplot(fig)

    
states_info=df['state'].value_counts()



sdf=pd.DataFrame()

sdf['state']=states_info.index

sdf['counts']=states_info.values



scl = [[0.0, 'rgb(247, 247, 117)'],[0.2, 'rgb(247, 188, 4)'],[0.4, 'rgb(247, 103, 4)'],\

            [0.6, 'rgb(247, 52, 4)'],[0.8, 'rgb(247, 4, 4)'],[1.0, 'rgb(94, 1, 1)']]



state_to_code = {'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME'}

sdf['state_code'] = sdf['state'].apply(lambda x : state_to_code[x])



data = [ dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = sdf['state_code'],

        z = sdf['counts'],

        locationmode = 'USA-states',

        text = sdf['state'],

        marker = dict(

            line = dict (

                color = 'white',

                width = 3

            ) ),

        colorbar = dict(

            title = "Gun Violence Incidents")

        ) ]



layout = dict(

        title = "Numbers of Gun Violence Incidents by State",

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'white'),

             )

fig = dict( data=data, layout=layout )

iplot( fig, filename='d3-cloropleth-map' )