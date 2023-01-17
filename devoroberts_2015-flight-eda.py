import pandas as pd 

import numpy as np 

import plotly

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

import matplotlib.pyplot as plt 

import seaborn as sns 

from datetime import datetime



sns.set(style="whitegrid")

# warnings.filterwarnings("ignore")

def load_datasets():

    airlines = pd.read_csv('../input/flight-delays/airlines.csv')

    airports = pd.read_csv('../input/flight-delays/airports.csv')

    flights = pd.read_csv('../input/flight-delays/flights.csv')

    return (airlines, airports, flights)



datasets = load_datasets()

airlines_df = datasets[0]

airports_df = datasets[1]

flights_df = datasets[2]

airlines_df.head()
print(f'Dataframe has {airlines_df.shape[0]} rows, and {airlines_df.shape[1]} columns.')



airports_df.head()
print(f'Dataframe has {airports_df.shape[0]} rows, and {airports_df.shape[1]} columns.')



weekday_dict = {

    1 : 'Monday',

    2 : 'Tuesday',

    3 : 'Wednesday',

    4 : 'Thursday',

    5 : 'Friday',

    6 : 'Saturday',

    7 : 'Sunday',

}



month_dict = {

    1 : 'Jan',

    2 : 'Feb',

    3 : 'Mar', 

    4 : 'Apr',

    5 : 'May',

    6 : 'Jun', 

    7 : 'Jul', 

    8 : 'Aug',

    9 : 'Sep',

    10 : 'Oct',

    11 : 'Nov',

    12 : 'Dec'

}



flights_df['DAY_OF_WEEK'] = flights_df['DAY_OF_WEEK'].map(weekday_dict)

flights_df['flight_date'] = [datetime(year, month, day) for year, month, day in zip(flights_df.YEAR, flights_df.MONTH, flights_df.DAY)]

flights_df['MONTH'] = flights_df['MONTH'].map(month_dict)

flights_df.head()
print(f'Dataframe has {flights_df.shape[0]} rows, and {flights_df.shape[1]} columns.')



# Rename airline code column.

airlines_df.rename(columns={'IATA_CODE':'AIRLINE_CODE'}, inplace=True)

# Rename airport code column.

airports_df.rename(columns={'IATA_CODE':'AIRPORT_CODE'}, inplace=True)

# Rename flights airline code column.

flights_df.rename(columns={'AIRLINE':'AIRLINE_CODE'}, inplace=True)

# Rename flights origin code column.

flights_df.rename(columns={'ORIGIN_AIRPORT':'ORIGIN_AIRPORT_CODE'}, inplace=True)

# Rename flights destination code column.

flights_df.rename(columns={'DESTINATION_AIRPORT':'DESTINATION_AIRPORT_CODE'}, inplace=True)





combined_df = pd.merge(flights_df, airlines_df, on='AIRLINE_CODE', how='left')

combined_df = pd.merge(combined_df, airports_df, left_on='ORIGIN_AIRPORT_CODE', right_on='AIRPORT_CODE', how='left')

combined_df = pd.merge(combined_df, airports_df, left_on='DESTINATION_AIRPORT_CODE', right_on='AIRPORT_CODE', how='left')



# Caculating flight airtime

combined_df['elapsed_time'] = combined_df['WHEELS_ON'] - combined_df['WHEELS_OFF']

combined_df.fillna(value=0.0, inplace=True)

combined_df.head()

# Rename origin airport meta columns.

combined_df.rename(columns={'AIRPORT_x':'ORIGIN_AIRPORT', 

                            'CITY_x':'ORIGIN_CITY', 

                            'STATE_x':'ORIGIN_STATE',

                            'COUNTRY_x':'ORIGIN_COUNTRY',

                            'LATITUDE_x':'ORIGIN_LATITUDE',

                            'LONGITUDE_x':'ORIGIN_LONGITUDE'}, inplace=True)

# Rename destination airport meta columns.

combined_df.rename(columns={'AIRPORT_y':'DESTINATION_AIRPORT', 

                            'CITY_y':'DESTINATION_CITY', 

                            'STATE_y':'DESTINATION_STATE',

                            'COUNTRY_y':'DESTINATION_COUNTRY',

                            'LATITUDE_y':'DESTINATION_LATITUDE',

                            'LONGITUDE_y':'DESTINATION_LONGITUDE'}, inplace=True)

number_of_flights = combined_df.shape[0]
origin_airport_group = combined_df.groupby('ORIGIN_AIRPORT')['FLIGHT_NUMBER'].count().sort_values(ascending=False)



destination_airport_group = combined_df.groupby('DESTINATION_AIRPORT')['FLIGHT_NUMBER'].count().sort_values(ascending=False)



airline_group = combined_df.groupby('AIRLINE')['FLIGHT_NUMBER'].count().sort_values(ascending=False)



labels = list(origin_airport_group[1:11].index)

values = list(origin_airport_group[1:11].values)



trace = go.Pie(labels=labels, values=values)

layout = go.Layout(title='Origin Airport Flight Distribution (Top 10)',

                    autosize=False,

                    width=800,

                    height=500,

                    margin=go.layout.Margin(

                        l=50,

                        r=50,

                        b=100,

                        t=100,

                        pad=4

                    ))



fig = go.Figure(data=[trace], layout=layout)

iplot(fig, filename='origin_distribution')



labels = list(destination_airport_group[1:11].index)

values = list(destination_airport_group[1:11].values)



trace = go.Pie(labels=labels, values=values)

layout = go.Layout(title='Destination Airport Flight Distribution',

                    autosize=False,

                    width=800,

                    height=500,

                    margin=go.layout.Margin(

                        l=50,

                        r=50,

                        b=100,

                        t=100,

                        pad=4

                    ))



fig = go.Figure(data=[trace], layout=layout)

iplot(fig, filename='destination_distribution')



labels = list(airline_group[:10].index)

values = list(airline_group[:10].values)



trace = go.Pie(labels=labels, values=values)

layout = go.Layout(title='Flight Distribution by Airline',

                    autosize=False,

                    width=800,

                    height=500,

                    margin=go.layout.Margin(

                        l=50,

                        r=50,

                        b=100,

                        t=100,

                        pad=4

                    ))



fig = go.Figure(data=[trace], layout=layout)

iplot(fig, filename='airline_distribution')







city_group = combined_df.groupby(['ORIGIN_CITY'])['FLIGHT_NUMBER'].count().sort_values(ascending=False)

city_group[1:21]



trace = go.Bar(x=city_group[1:21].index, 

                y=city_group[1:21].values, 

                name='city',

                marker={

                    'color':city_group[1:21].values,

                    'colorscale':'Reds',

                    'showscale':True,

                    },

                )



layout = go.Layout(title='Number of Flights from Origin City',

                    xaxis={'title':'Origin City'},

                    yaxis={'title':'# of Flights'},

                    autosize=False,

                    width=800,

                    height=500,

                    margin=go.layout.Margin(

                        l=50,

                        r=50,

                        b=100,

                        t=100,

                        pad=4

                    ))



fig = go.Figure(data=[trace], layout=layout)

iplot(fig, filename='origin_city_bar')





trace = go.Bar(x=airline_group[:10].index, 

                y=airline_group[:10].values, 

                name='airlines',

                marker={

                    'color':airline_group[:10].values,

                    'colorscale':'Reds',

                    'showscale':True,

                    },

                )



layout = go.Layout(title='Number of Flights by Airline',

                    xaxis={'title':'Airline'},

                    yaxis={'title':'# of Flights'},

                    autosize=False,

                    width=800,

                    height=500,

                    margin=go.layout.Margin(

                        l=50,

                        r=50,

                        b=100,

                        t=100,

                        pad=4

                    ))



fig = go.Figure(data=[trace], layout=layout)

iplot(fig, filename='airline_bar')



month_group = combined_df.groupby(['MONTH'])['FLIGHT_NUMBER'].count()



trace = go.Bar(x=month_group.index, 

                y=month_group.values, 

                name='month',

                marker={

                    'color':month_group.values,

                    'colorscale':'Reds',

                    'showscale':True,

                    },

                )



layout = go.Layout(title='Number of Flights per Month',

                    xaxis={'title':'Month'},

                    yaxis={'title':'# of Flights'},

                    autosize=False,

                    width=500,

                    height=500,

                    margin=go.layout.Margin(

                        l=50,

                        r=50,

                        b=100,

                        t=100,

                        pad=4

                    )

                )



fig = go.Figure(data=[trace], layout=layout)

iplot(fig, filename='month_bar')



day_group = combined_df.groupby(['DAY_OF_WEEK'])['FLIGHT_NUMBER'].count().sort_values(ascending=False)



trace = go.Bar(x=day_group.index, 

                y=day_group.values, 

                name='day_of_week',

                marker={

                    'color':day_group.values,

                    'colorscale':'Reds',

                    'showscale':True,

                    },

                )



layout = go.Layout(title='Number of Flights per Day Of Week',

                    xaxis={'title':'Day Of Week'},

                    yaxis={'title':'# of Flights'},

                    autosize=False,

                    width=500,

                    height=500,

                    margin=go.layout.Margin(

                        l=50,

                        r=50,

                        b=100,

                        t=100,

                        pad=4

                    )

                )



fig = go.Figure(data=[trace], layout=layout)

iplot(fig, filename='day_bar')



# flights_df.head(30)