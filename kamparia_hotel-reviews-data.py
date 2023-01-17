# imports pandas, numpy and matplotlib modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# import plotly modules
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')
## load csv data
file_path = '../input/7282_1.csv'
df = pd.read_csv(file_path)
## rename column names
old_names = ['reviews.date', 'reviews.rating', 'reviews.title', 'reviews.text'] 
new_names = ['date', 'rating', 'title', 'text']
df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
## subset multiple columns of our dataframe
df = df[['latitude', 'longitude', 'name', 'address', 'postalCode', 'categories', 'city', 'country', 'date', 'rating', 'title', 'text',]] 
## drop rows with NAs'
df = df[pd.notnull(df['name'])]
df = df[pd.notnull(df['latitude'])]
df = df[pd.notnull(df['longitude'])]
df = df[pd.notnull(df['rating'])]
df = df[pd.notnull(df['date'])]
# Q1. Which hotel has the highest number of reviews.
q1 = df['name'].value_counts().reset_index().iloc[0]['index']
print("Answer: " + q1)
# Q2. Which hotel has the highest average rating of reviews.
q2 = df.groupby('name')['rating'].mean().reset_index().sort_values(by='rating', ascending=False).iloc[0]['name']
print("Answer: " + q2)
q2 = df.groupby('name')['rating'].mean().reset_index().sort_values(by='rating', ascending=False)[:10]
trace = go.Bar(
    x=q2['name'],
    y=q2['rating'],
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),    
    opacity=0.6
)

data = [trace]

layout = go.Layout(
    title='Bar Chat Showing Top 10 Hotels With Highest Average Ratings.',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='hotel-reviews-highest-rating')
# Q3. Which City has the highest number of hotels.
q3 = df['city'].value_counts().reset_index().iloc[0]['index']
print("Answer: " + q3)
q3 = df['city'].value_counts()[:20]
trace = go.Bar(
    x=q3.index,
    y=q3.values,
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),    
    opacity=0.6
)

data = [trace]

layout = go.Layout(
    title='Bar Chart Showing Top 20 Cities With Highest Reviews.',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='hotel-reviews-highest-cities')
# Q4. What is the relationship between total number of reviews per hotel and average rating of the hotel.
group_name = df.groupby(['name'])['rating'].mean().reset_index()

group_count = df.groupby('name').count().reset_index()
old_names = ['latitude'] 
new_names = ['count']
group_count.rename(columns=dict(zip(old_names, new_names)), inplace=True)
group_count = group_count[['name', 'count']] 

q4 = pd.merge(group_name, group_count, left_index=True, right_index=True)[['name_x', 'rating', 'count']] 

# q4.plot.scatter(x='rating', y='count')

x = (q4['rating']).values
y = (q4['count']).values

data = go.Data([
    go.Scatter(
        x = x,
        y = y,
        mode = 'markers'
    )
])

layout = go.Layout(
    title='Diagram Showing The Relationship Between Ratings & Number Of Reviews.',
)

fig = dict(data=data, layout=layout)

# Plot and embed in ipython notebook!
py.iplot(fig, filename='hotels-reviews-scatter')

print("Answer: Based on the above scatter plot diagram, there is no relationship between the average rating of an hotel and the total number of reviews of the hotel.")
# Q5. Plot an interactive map of the hotels and average review ratings as a label.
q5 = df.groupby(['name', 'latitude', 'longitude'])['rating'].mean().reset_index()
lat = q5.latitude
lon = q5.longitude
name = q5.name
rating = round(q5.rating,2)

mapbox_access_token = 'pk.eyJ1Ijoia2FtcGFyaWEiLCJhIjoib0JLTExtSSJ9.6ahf835RV3kBUnC3cQ-SnA'
data = go.Data([
    go.Scattermapbox(
        lat=lat,
        lon=lon,
        mode='markers',
        marker=go.Marker(
            size=10,
            color='rgb(255, 0, 0)',
            opacity=0.7
        ),
        text=rating,
        hoverinfo=''
    ),
    go.Scattermapbox(
        lat=lat,
        lon=lon,
        mode='markers',
        marker=go.Marker(
            size=8,
            color='rgb(242, 177, 172)',
            opacity=0.7
        ),
        text=rating,
        hoverinfo=''
    )]
)
        
layout = go.Layout(
    title='Interactive Map Showing The Location Of Hotels & Average Ratings.',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=38,
            lon=-94
        ),
        pitch=0,
        zoom=3,
        style='dark'
    ),
)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='hotel-reviews-map')