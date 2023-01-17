import pandas as pd

import numpy as np

import plotly.graph_objects as go

import plotly.express as px
df = pd.read_csv("../input/au-dom-traffic/audomcitypairs-201909.csv")
df.head()
passenger_load = []

for i in df['City1'].unique():

    passenger_load.append([i,df[df['City1']==i].groupby('City1')['Passenger_Load_Factor'].mean()[0]])

passenger_load = pd.DataFrame(passenger_load)

passenger_load.columns = ["City","PassengerLoadFactor"]
fig = go.Figure(data=[go.Bar(x=passenger_load['City'],y=passenger_load['PassengerLoadFactor'],hovertext=passenger_load['City'])],layout_title_text="Average Number oF Passenger Load Factor for City")

fig.show()

# px.bar(x='City',y='PassengerLoadFactor',data_frame=passenger_load)
px.scatter(x='Passenger_Load_Factor',y='City1',data_frame=df,title="Load Factor of Each City")
px.scatter(x='Aircraft_Trips',y='City1',data_frame=df)
_20191 = df[(df['Year']==2019)&(df['Month_num']==1)]

_20191.head()
average_seats = _20191.groupby('City1')['Seats'].mean()

px.bar(data_frame=df,x=average_seats.index,y=average_seats.values)
groupbyCity = _20191.groupby('City1')['Distance_GC_(km)'].max()

px.line(data_frame=groupbyCity,x=groupbyCity.index,y=groupbyCity.values)
year_passenger_trip = df.groupby('Year')['Passenger_Trips'].mean()

px.line(data_frame=year_passenger_trip,x=year_passenger_trip.index,y=year_passenger_trip.values,)
year_passenger_load_factor = df.groupby('Year')['Passenger_Load_Factor'].mean()

px.line(x=year_passenger_load_factor.index,y=year_passenger_load_factor.values)
df.head()
distance_year_min = df.groupby('Year')['Distance_GC_(km)'].mean().sort_values()
px.bar(x=distance_year_min.index,y=distance_year.values)
_2019Seats = df[df['Year']==2019].groupby('Month_num')['Seats'].mean()

px.line(x=_2019Seats.index,y=_2019Seats.values)
average_distance_mean = df.groupby('Year')['Distance_GC_(km)'].mean()

px.line(average_distance_mean.index,average_distance_mean.values)