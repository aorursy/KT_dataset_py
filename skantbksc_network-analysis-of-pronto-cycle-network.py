# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import igraph as ig

import matplotlib.pyplot as plt

import random

import datetime as dt

import seaborn as sns

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster



from patsy import dmatrices

import statsmodels.api as sm

trip = pd.read_csv('/kaggle/input/cycle-share-dataset/trip.csv',parse_dates=['starttime','stoptime'], error_bad_lines=False)

station = pd.read_csv('/kaggle/input/cycle-share-dataset/station.csv', error_bad_lines=False)

weather = pd.read_csv('/kaggle/input/cycle-share-dataset/weather.csv', parse_dates=['Date'], error_bad_lines=False)
# Create a base map

m_4 = folium.Map(width=500,height=500,

                 location=[47.642394, -122.323738], 

                 tiles='openstreetmap',

                 zoom_start=12,

                 min_zoom=12,max_zoom=12

                )



# Add a bubble map to the base map

for i in range(0,len(station)):

    Circle(

        location=[station.iloc[i]['lat'], station.iloc[i]['long']],

        radius=20,

        popup=station.iloc[i]['station_id'],

#         color=color_producer(loc.iloc[i]['usage'])

    ).add_to(m_4)





m_4
trip['month'] = trip['starttime'].dt.strftime('%Y-%m')

trip['quarter'] = trip['starttime'].dt.to_period('Q')

trip
# Grouping the trips data by month

trip_monthly = trip.groupby(['month']).agg(

    nbr_of_trips = pd.NamedAgg(column = 'trip_id',aggfunc = 'count')).reset_index()



trip_monthly

plt.figure(figsize=(10,4))

plt.plot(trip_monthly['month'],trip_monthly['nbr_of_trips'])

plt.xticks(rotation='vertical')

plt.ylabel('nbr of trips')

plt.show()
weather['month'] = weather['Date'].dt.strftime('%Y-%m')

rainy_weather = weather[weather['Events']=='Rain']

rainy_weather_count = rainy_weather[['month','Events']].groupby(['month']).count().reset_index()



plt.figure(figsize=(10,4))

plt.plot(rainy_weather_count['month'],rainy_weather_count['Events'])

plt.xticks(rotation='vertical')

plt.ylabel('nbr of rainy days')

plt.show()
# Grouping the trips by usertype

trip_usertype = trip.groupby(['month','usertype']).agg(

    nbr_of_trips = pd.NamedAgg(column = 'trip_id',aggfunc = 'count')).reset_index()



pd.pivot(trip_usertype, values='nbr_of_trips', index='month',columns='usertype')
plt.figure(figsize=(10,6))

sns.lineplot(x='month', y='nbr_of_trips', hue='usertype', data=trip_usertype)

plt.xticks(rotation=45)

plt.show()
# Grouping the trips data by gender

trip_gender = trip.groupby(['month','gender']).agg(

    nbr_of_trips = pd.NamedAgg(column = 'trip_id',aggfunc = 'count')).reset_index()



pd.pivot(trip_gender, values='nbr_of_trips', index='month',columns='gender')
plt.figure(figsize=(10,6))

sns.lineplot(x='month', y='nbr_of_trips', hue='gender', data=trip_gender)

plt.xticks(rotation=45)

plt.show()
df = trip.groupby(['from_station_id','to_station_id']).agg(

    nbr_of_trips = pd.NamedAgg(column = 'trip_id',aggfunc = 'count')).reset_index()

df
# We construct the network with stations as the node and number of trips as edges

G = ig.Graph.TupleList(

    df[['from_station_id','to_station_id','nbr_of_trips']].itertuples(index=False),

    directed = True, 

    edge_attrs="nbr_of_trips"

)



# Plot of the network

ig.plot(

    G, layout=G.layout("kk"), bbox = (500,500),

    vertex_label_size = 4, vertex_size = 10

)
# Random deletion of nodes

def random_deletion(graph):

    g = graph.copy()

    # Generate the random sequence of nodes to be deleted

    g0 = len(g.vs)



    random.seed(0)

    del_index = random.sample(range(g0), g0)

    del_seq = [g.vs[del_index[i]]['name'] for i in del_index]



    # Perform deletion one by one. If a node results in splitting of network into disconnected subgraphs, 

    # the larger subgraph is considered for further calculations.



    nbr_of_del_nodes=[0]

    diameter = [g.diameter(directed=True,unconn=True,weights=None)]

    avg_path_len = [g.average_path_length(directed=True, unconn=True)]

    density = [g.density(loops=False)]





    for i in range(g0):

        try:        

            g.delete_vertices([del_seq[i]])

            dia = g.diameter(directed=True, unconn=True, weights=None)

            diameter.append(dia)

            path_len = g.average_path_length(directed=True, unconn=True)

            avg_path_len.append(path_len)

            dens = g.density(loops=False)

            density.append(dens) 

            nbr_of_del_nodes.append(g0 - len(g.vs))

        except:

            pass



    results = pd.DataFrame({'nbr_of_del_nodes': nbr_of_del_nodes,

                            'diameter': diameter,

                            'avg_path_len': avg_path_len,

                            'density': density})



    return(results)

random_deletion(G)
# Sequential deletion of nodes 

def sequential_deletion(graph):



    g = graph.copy()

    # Obtain the degree of all nodes

    degree_df = pd.DataFrame({'vertex':g.vs.indices,

                  'station_id': g.vs['name'],

                  'degree':g.degree(g.vs.indices, mode='ALL', loops=False)}).sort_values(by=['degree'],ascending=False)

    del_seq = degree_df['station_id'].tolist()



    g0 = len(g.vs)

    diameter = [g.diameter(directed=True, unconn=True, weights=None)]

    nbr_of_del_nodes = [0]

    avg_path_len = [g.average_path_length(directed=True, unconn=True)]

    density = [g.density(loops=False)]



    for i in range(g0):

        try:

            g.delete_vertices([del_seq[i]])

            dia = g.diameter(directed=True, unconn=True, weights=None)

            diameter.append(dia)

            path_len = g.average_path_length(directed=True, unconn=True)

            avg_path_len.append(path_len)

            dens = g.density(loops=False)

            density.append(dens)

            nbr_of_del_nodes.append(g0-len(g.vs))

        except:

            pass

    results = pd.DataFrame({'nbr_of_del_nodes': nbr_of_del_nodes,

                             'diameter': diameter,

                             'avg_path_len': avg_path_len,

                             'density': density})



    return(results)
sequential_deletion(G)
# Plotting of charts between random and sequential deletion

def rand_and_seq_del_charts(graph):

    results1 = random_deletion(graph)

    results2 = sequential_deletion(graph)



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6),sharey=True)

    fig.suptitle('Comparison between random and sequential deletion of nodes', fontsize=20)

    ax1.plot(results1.index,results1.avg_path_len, color='b', label='avg path length')

    ax1.plot(results1.index,results1.diameter, color='r', label='diameter')

    ax1.plot(results1.index,results1.density, color='g', label='density')

    ax1.legend(loc="upper right")

    ax1.set_xlabel("number of nodes deleted")

    ax1.set_title("deletion with random sequence")





    ax2.plot(results2.index,results2.avg_path_len, color='b', label='avg path length')

    ax2.plot(results2.index,results2.diameter, color='r', label='diameter')

    ax2.plot(results2.index,results2.density, color='g', label='density')

    ax2.legend(loc="upper right")

    ax2.set_xlabel("number of nodes deleted")

    ax2.set_title("deletion sequence with degree importance")

    # plt.tight_layout()

    plt.show()

    pass
rand_and_seq_del_charts(G)
# Grouping the trips data by day

trip['Date'] = trip['starttime'].dt.date

trip_daily = trip.groupby(['Date','from_station_id','to_station_id']).agg(

    nbr_of_trips = pd.NamedAgg(column = 'trip_id',aggfunc = 'count')).reset_index()

trip_daily
date_list = trip_daily['Date'].unique()



diameter = []

avg_path_len = []

density = []



for date in date_list:

    df = trip_daily[trip_daily['Date']== date]

    g = ig.Graph.TupleList(

        df[['from_station_id','to_station_id','nbr_of_trips']].itertuples(index=False),

        directed = True, 

        edge_attrs="nbr_of_trips")

    dia = g.diameter(directed=True, unconn=True, weights=None)

    diameter.append(dia)

    path_len = g.average_path_length(directed=True, unconn=True)

    avg_path_len.append(path_len)

    dens = g.density(loops=True)  

    density.append(dens)
daily_network_features = pd.DataFrame({'Date': date_list,

                                      'Diameter': diameter,

                                      'Avg_path_len': avg_path_len,

                                      'Density': density})

daily_network_features
# Next we consider the weather parameters as the regressors. We take mean temperature, mean humidity, mean wind speed and precipitation as regressors.

weather
# Prepare the regressors dataframe

weather_df = weather.copy()

weather_df['Date'] = weather_df['Date'].dt.date

weather_df['Events'] = weather_df['Events'].fillna('Sunny')

weather_df = weather_df.fillna(method='ffill')

weather_df
dataset = pd.merge(daily_network_features, weather_df, on = 'Date')

trip_daily = trip.groupby(['Date']).agg(

    nbr_of_trips = pd.NamedAgg(column = 'trip_id',aggfunc = 'count')).reset_index()

dataset =pd.merge(trip_daily, dataset, on = 'Date')

dataset
y, X = dmatrices('Avg_path_len ~ Mean_Temperature_F + Mean_Wind_Speed_MPH + Precipitation_In', data=dataset, return_type='dataframe')



ols_model=sm.OLS(y,X)

result=ols_model.fit()

print(result.summary2())
y, X = dmatrices('nbr_of_trips ~ Mean_Temperature_F + Mean_Wind_Speed_MPH + Precipitation_In', data=dataset, return_type='dataframe')



ols_model=sm.OLS(y,X)

result=ols_model.fit()

print(result.summary2())