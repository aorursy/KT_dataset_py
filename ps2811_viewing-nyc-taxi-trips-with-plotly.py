#importing packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

import time

import networkx as nx

from plotly.graph_objs import *
#Load the data

train = pd.read_csv('../input/new-york-city-taxi-with-osrm/train.csv')

test = pd.read_csv('../input/new-york-city-taxi-with-osrm/test.csv') 

train_NearestCities = pd.read_csv('../input/nearest-cities-for-nyc-taxi-trips/train_NearestCities.csv')

test_NearestCities = pd.read_csv('../input/nearest-cities-for-nyc-taxi-trips/test_NearestCities.csv')
#Extract DateTime features

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

                                         

train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date

test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date



train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday

train.loc[:, 'pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear

train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour

train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute

train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()



test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday

test.loc[:, 'pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear

test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour

test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute

test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()
scatter_data = train[train.trip_duration > 3600]  #get trips for more than an hour to avoid plotting bigdata.

x_values = scatter_data['pickup_hour']

y_values = scatter_data['trip_duration']/3600



trace = go.Scatter(x = x_values,y = y_values,mode = 'markers')

scatdata = [trace]

scatdata2 = dict(data=scatdata,layout=dict(title='Trip Duration(Hrs) [Filtered for trips that took more than an hour]'))

#Plot and embed in ipython notebook!

py.iplot(scatdata2, filename='tripduration-scatter')
#h0 = train.loc[train['pickup_hour'] == 0, 'trip_duration']

#Remove the four outlier durations

subset1 = train.loc[train['trip_duration'] < 250000]

#Lets convert trip duration to minutes.

subset1['trip_duration_mins'] = subset1['trip_duration']/60

data = []

for pick in range(0,23):

    data.append(go.Box( y=subset1.loc[subset1['pickup_hour'] == pick, 'trip_duration_mins'], name = pick, showlegend = False))   



layout = go.Layout(title = "Box Plot with Outlier trip durations")



fig = go.Figure(data=data,layout=layout)

py.iplot(fig, filename='tripduration-box-plot')
#Data Preparation for the weekly analysis

#get count of trips every hour on every weekday.

sunday = train[train['pickup_weekday'] == 6]

df_sundayhourlytripcount = sunday.groupby('pickup_hour').count()

monday = train[train['pickup_weekday'] == 0]

df_mondayhourlytripcount = monday.groupby('pickup_hour').count()

tuesday = train[train['pickup_weekday'] == 1]

df_tuesdayhourlytripcount = tuesday.groupby('pickup_hour').count()

wednesday = train[train['pickup_weekday'] == 2]

df_wednesdayhourlytripcount = wednesday.groupby('pickup_hour').count()

thursday = train[train['pickup_weekday'] == 3]

df_thursdayhourlytripcount = thursday.groupby('pickup_hour').count()

friday = train[train['pickup_weekday'] == 4]

df_fridayhourlytripcount = friday.groupby('pickup_hour').count()

saturday = train[train['pickup_weekday'] == 5]

df_saturdayhourlytripcount = saturday.groupby('pickup_hour').count()

pickuphr_x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

sun_tripcounty = df_sundayhourlytripcount['id']

mon_tripcounty = df_mondayhourlytripcount['id']

tues_tripcounty = df_tuesdayhourlytripcount['id']

wed_tripcounty = df_wednesdayhourlytripcount['id']

thurs_tripcounty = df_thursdayhourlytripcount['id']

fri_tripcounty = df_fridayhourlytripcount['id']

sat_tripcounty = df_saturdayhourlytripcount['id']

# Create traces

trace1 = go.Scatter(x = pickuphr_x,y = sun_tripcounty,mode = 'Sunday',name = 'Sunday')

trace2 = go.Scatter(x = pickuphr_x, y = mon_tripcounty, mode = 'Monday', name = 'Monday')

trace3 = go.Scatter( x = pickuphr_x, y = tues_tripcounty,  mode = 'Tuesday', name = 'Tuesday')

trace4 = go.Scatter( x = pickuphr_x, y = wed_tripcounty, mode = 'Wednesday', name = 'Wednesday')

trace5 = go.Scatter( x = pickuphr_x, y = thurs_tripcounty, mode = 'Thursday',  name = 'Thursday')

trace6 = go.Scatter( x = pickuphr_x, y = fri_tripcounty, mode = 'Friday',  name = 'Friday')

trace7 = go.Scatter(x = pickuphr_x, y = sat_tripcounty,  mode = 'Saturday',  name = 'Saturday')

layout = dict(title = 'Weekly Trip Demand by Hour')

linedata = [trace1, trace2, trace3, trace4, trace5, trace6,trace7]

fig = dict(data=linedata, layout=layout)

py.iplot(fig, filename='timeline-lineplot')
#Preparing data for the graph

W_0 = train[train['pickup_weekday'] == 0].groupby('pickup_hour').count()

W_1 = train[train['pickup_weekday'] == 1].groupby('pickup_hour').count()

W_2 = train[train['pickup_weekday'] == 2].groupby('pickup_hour').count()

W_3 = train[train['pickup_weekday'] == 3].groupby('pickup_hour').count()

W_4 = train[train['pickup_weekday'] == 4].groupby('pickup_hour').count()

W_5 = train[train['pickup_weekday'] == 5].groupby('pickup_hour').count()

W_6 = train[train['pickup_weekday'] == 6].groupby('pickup_hour').count()

trace = go.Heatmap(z=[W_0.id,W_1.id,W_2.id,W_3.id,W_4.id,W_5.id,W_6.id],

                    y=['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday'],

                   x=['Midnight','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','Noon','1pm','2pm',

                     '3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'],

                  colorscale='Electric',xgap = 10,ygap = 10,)



layout = dict(title = 'Trips per week per hour')

dataheat=[trace]

fig = dict(data = dataheat, layout=layout)

py.iplot(fig, filename='labelled-heatmap')
#Distance Feature

#(From Beluga's kernel') Calculating Manhattan Distance/direction)

def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



def dummy_manhattan_distance(lat1, lng1, lat2, lng2):

    a = haversine_array(lat1, lng1, lat1, lng2)

    b = haversine_array(lat1, lng1, lat2, lng1)

    return a + b



def bearing_array(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))



train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)



train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2

train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2

train.loc[:, 'avg_speed_hr'] = 1000 * train['distance_haversine'] / train['trip_duration']

data = [go.Scatter(

          x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],

          y=train.groupby('pickup_hour').mean()['avg_speed_hr'])]

layout = dict(title = 'Average speed(m/s) at every hour')

fig2 = dict(data = data, layout=layout)

py.iplot(fig2, filename='labelled-heatmap')
#For this visual joined the train dataset with NearestCities dataset. 

#For clarity, let us plot only the trip records for one month.

train["month"] = train['pickup_datetime'].dt.month



#Lets first merge the two datasets

merged = pd.merge(train,train_NearestCities, on='id', how='left')

train_jan = merged[merged['month'] == 1]

merged_sub = merged.head(50000)
#So actually we will have all possible cities as nodes. 

#We need node positions - let take random numbers in pairs x and y for it. 

#Finding the list of cities to be plotted on network.

pickup = merged_sub['Nearest_PickupCity'].tolist()

dropoff = merged_sub['Nearest_DropoffCity'].tolist()



citylist = pickup + dropoff

finalcity = set(citylist)

finallist = list(finalcity)



len(finallist)
#Now generating nodes randomly

G=nx.random_geometric_graph(85,0.125)

pos=nx.get_node_attributes(G,'pos')



dmin=1

ncenter=0

for n in pos:

    x,y=pos[n]

    d=(x-0.5)**2+(y-0.5)**2

    if d<dmin:

        ncenter=n

        dmin=d



p=nx.single_source_shortest_path_length(G,ncenter)
#Generating node trace

node_trace = Scatter(

    x=[],

    y=[],

    text=[],

    mode='markers',

    hoverinfo='text',

    marker=Marker(

        showscale=True,

        colorscale='YIOrRd',

        reversescale=True,

        color=[],

        size=16,

        colorbar=dict(

            thickness=15,

            title='Node Connections',

            xanchor='left',

            titleside='right'

        ),

        line=dict(width=2)))



for node in G.nodes():

    x, y = G.node[node]['pos']

    node_trace['x'].append(x)

    node_trace['y'].append(y)
#Assigning random locations to citites

city_random_x = node_trace.x

city_random_y = node_trace.y



#Remove the random edges created by default 

G.remove_edges_from(G.edges())
#City random co-ordinates #Add node points for edges.

city_coords = pd.DataFrame({'City': finallist,'node_x': city_random_x,'node_y': city_random_y, 'City_NodeNumber': G.nodes()})

city_coords.head()
#Now make edges

#Get coords for Pickup/Dropoff City

df2 = pd.merge(merged_sub, city_coords,how='left', left_on=['Nearest_PickupCity'], right_on=['City'])

df2.rename(columns={'node_x' : 'Pickup_node_x','node_y' : 'Pickup_node_y','City_NodeNumber' : 'PickupNode'}, inplace=True)

#Now add drop off city corrds for edges

df3 = pd.merge(df2, city_coords,how='left', left_on=['Nearest_DropoffCity'], right_on=['City'])

df3.rename(columns={'node_x' : 'Dropoff_node_x','node_y' : 'Dropoff_node_y','City_NodeNumber' : 'DropoffNode'}, inplace=True)

df3['edge_pairs'] = list(zip(df3.PickupNode, df3.DropoffNode))
#Creating edge trace based on random locations

graph = list(set(df3.edge_pairs))

for edge in graph:

    G.add_edge(edge[0], edge[1])

    

edge_trace = Scatter(

    x=[],

    y=[],

    line=Line(width=0.5,color='#CCC'),

    hoverinfo='none',

    mode='lines')
#Adding edges based on pickup dropff city pairs or taxi trips.

for index, row in df3.iterrows():

    edge_trace['x'] += [row['Pickup_node_x'], row['Dropoff_node_x'], None]

    edge_trace['y'] += [row['Pickup_node_y'], row['Dropoff_node_y'], None]
#Assign a node number to each city for plotting

nodes_dict = dict(zip(city_coords.City,city_coords.City_NodeNumber))

labels = nodes_dict  

for node in G.nodes():

    labels[node] = node
#Getting adjacent nodes

i =0

for node, adjacencies in enumerate(G.adjacency_list()):

    node_trace['marker']['color'].append(len(adjacencies))

    node_info = str(finallist[i]) +'| # of Unique City trips: '+str(len(adjacencies)) 

    #print(node_info)

    node_trace['text'].append(node_info)

    #print(node_trace['text'][i])

    i = i + 1
fig = Figure(data=Data([edge_trace, node_trace]),

             layout=Layout(

                title='<br># of Unique Trip Routes across and within cities Using Networkx',

                titlefont=dict(size=16),

                showlegend=False,

                hovermode='closest',

                margin=dict(b=20,l=5,r=5,t=40),

                annotations=[ dict(

                    text="In the graph: Node = City|Edges = Cross City Trips| Node Color = # of trips per city",

                    showarrow=False,

                    xref="paper", yref="paper",

                    x=0.005, y=-0.002 ) ],

                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),

                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))



py.iplot(fig, filename='networkx-taxitrips')
#Lets look at demand per hour.

bar1data = merged[merged['Nearest_PickupCity']=='Manhattan'].groupby(['pickup_hour']).count().reset_index()

bar3data = merged[merged['Nearest_PickupCity']=='Long Island City'].groupby(['pickup_hour']).count().reset_index()

bar2data = merged[merged['Nearest_PickupCity']=='New York City'].groupby(['pickup_hour']).count().reset_index()
trace0 = go.Bar(

    x=['Midnight','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','Noon','1pm','2pm',

                     '3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'],

    y= bar1data['id'],

    name='Manhattan',

    marker=dict(color='rgb(49,130,189)' ))

trace1 = go.Bar(

    x=['Midnight','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','Noon','1pm','2pm',

                     '3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'],

    y= bar2data['id'],

    name='New York City',

    marker=dict(color='rgb(120,70,100)',))



trace2 = go.Bar(

    x=['Midnight','1am','2am','3am','4am','5am','6am','7am','8am','9am','10am','11am','Noon','1pm','2pm',

                     '3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'],

    y= bar3data['id'],

    name='Long Island City',

    marker=dict(color='rgb(130,200,110)',))

datax = [trace0, trace1, trace2]

layout = go.Layout(xaxis=dict(tickangle=-45),barmode='group',)



fig = go.Figure(data=datax, layout=layout)

py.iplot(fig, filename='City-BarPlot')