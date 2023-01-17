import numpy as np

import pandas as pd 

import plotly.offline as pyo

import plotly.plotly as py

from plotly.graph_objs import *

pyo.offline.init_notebook_mode()

import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns = 50

import plotly.plotly as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/BrFlights2/BrFlights2.csv', encoding='latin1')

df.head()
df.columns = ['Flights', 'Airline', 'Flight_Type','Departure_Estimate','Departure_Real','Arrival_Estimate','Arrival_Real','Flight_Situation','Code_Justification','Origin_Airport','Origin_City','Origin_State','Origin_Country','Destination_Airport','Destination_City','Destination_State','Destination_Country','Destination_Long','Destination_Lat','Origin_Long','Origin_Lat']

print('Shape:',df.shape)

#__________________________________________

# info on variable types and filling factor

tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})

tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values'}))

tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.

                         rename(index={0:'null values (%)'}))

tab_info
AirlineFlights_Sorted = pd.DataFrame(df.groupby(by=['Flights','Airline'])['Departure_Estimate'].count().reset_index())

AirlineFlights_Sorted.columns = ['Flights','Airline','Count']

AirlineFlights_Sorted = AirlineFlights_Sorted.sort_values(by='Count',ascending=False)

data=[]

AirlineFlights_Sorted_Top=AirlineFlights_Sorted.iloc[:20,:]

for Airname in list(AirlineFlights_Sorted_Top['Airline'].unique()):

    data.append(go.Bar(

    x = list(AirlineFlights_Sorted_Top[AirlineFlights_Sorted_Top['Airline']==Airname]['Flights']),

    y = list(AirlineFlights_Sorted_Top[AirlineFlights_Sorted_Top['Airline']==Airname]['Count']),

    name= Airname

    )  

    )

layout = go.Layout(

    barmode='group',

    title = 'Count of 20 Most Used Flights'

)

fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)
AirlineType_Sorted = pd.DataFrame(df.groupby(by=['Flight_Type','Airline'])['Departure_Estimate'].count().reset_index())

AirlineType_Sorted.columns = ['Flight_Type','Airline','Count']

AirlineType_Sorted = AirlineType_Sorted.sort_values(by='Count',ascending=False)

trace1 = go.Pie(labels=list(AirlineType_Sorted[AirlineType_Sorted['Flight_Type']=='Nacional']['Airline']),

            values=list(AirlineType_Sorted[AirlineType_Sorted['Flight_Type']=='Nacional']['Count']),

            domain = {"x": [0.67, 1]},

            name = 'Nacional',

            hoverinfo="label+percent+name",

            hole= .4)

trace2 = go.Pie(labels=list(AirlineType_Sorted[AirlineType_Sorted['Flight_Type']=='Regional']['Airline']),

            values=list(AirlineType_Sorted[AirlineType_Sorted['Flight_Type']=='Regional']['Count']),

            domain = {"x": [0.34, 0.66]},

            name = 'Regional',

            hoverinfo="label+percent+name",

            hole= .4)





trace3 = go.Pie(labels=list(AirlineType_Sorted[AirlineType_Sorted['Flight_Type']=='Internacional'].iloc[:20,:]['Airline']),

            values=list(AirlineType_Sorted[AirlineType_Sorted['Flight_Type']=='Internacional'].iloc[:20,:]['Count']),

            domain = {"x": [0, 0.33]},

            name = 'Internacional',

            hoverinfo="label+percent+name",

            hole= .4)

layout = go.Layout(

    title="Distributions for Different Flight Types",

    showlegend = False,

    annotations = [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Nacional",

                "x": 0.9,

                "y": 0.5

            },

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Regional",

                "x": 0.5,

                "y": 0.5

            },

            {

                "font": {

                    "size": 14

                },

                "showarrow": False,

                "text": "Internacional",

                "x": 0.1,

                "y": 0.5

            }

        ]

)

fig = go.Figure(data=[trace1,trace2,trace3], layout=layout)

pyo.iplot(fig)
AirportFlights_Sorted = pd.DataFrame(df.groupby(by=['Origin_Airport','Flight_Type'])['Departure_Estimate'].count().reset_index())

AirportFlights_Sorted.columns = ['Origin_Airport','Flight_Type','Count']

AirportFlights_Sorted = AirportFlights_Sorted.sort_values(by='Count',ascending=False)

data=[]

AirportFlights_Sorted_Top=AirportFlights_Sorted.iloc[:20,:]

for Type in list(AirportFlights_Sorted_Top['Flight_Type'].unique()):

    data.append(go.Bar(

    x = list(AirportFlights_Sorted_Top[AirportFlights_Sorted_Top['Flight_Type']==Type]['Origin_Airport']),

    y = list(AirportFlights_Sorted_Top[AirportFlights_Sorted_Top['Flight_Type']==Type]['Count']),

    name= Type

    )  

    )

layout = go.Layout(

    barmode='group',

    title = 'Count of 20 Most Used Airports'

)

fig = go.Figure(data=data, layout=layout)

pyo.iplot(fig)
df_airports = pd.DataFrame()

df_airports['name'] = df['Origin_Airport'].unique()

df_airports['Lon'] = df['Origin_Long'].unique()

df_airports['Lat'] = df['Origin_Lat'].unique()

df_path = df[['Destination_Long', 'Destination_Lat','Origin_Long','Origin_Lat']]

df_path = df_path.drop_duplicates()

df_path = df_path.reset_index()

AirPortCount = pd.DataFrame(df.groupby(by=['Origin_Airport'])['Departure_Estimate'].count().reset_index())

AirPortCount.columns = ['Origin_Airport','Count']

df_airports = df_airports.merge(AirPortCount,left_on='name',right_on='Origin_Airport')

airports = [ dict(

        type = 'scattergeo',

        lon = df_airports['Lon'],

        lat = df_airports['Lat'],

        hoverinfo = 'text',

        text = df_airports['name'],

        mode = 'markers',

        marker = dict( 

            size=np.log10(df_airports['Count'])*1.35, 

            color='rgb(255, 0, 0)',

            line = dict(

                width=3,

                color='rgba(68, 68, 68, 0)'

            )

        ))]

flight_paths = []

for i in range(len(df_path)):

    flight_paths.append(dict(

            type = 'scattergeo',

            lon = [df_path['Origin_Long'][i], df_path['Destination_Long'][i]],

            lat = [df_path['Origin_Lat'][i], df_path['Destination_Lat'][i]],

            mode = 'lines',

            line = dict(

                width = 0.1,

                color = 'red',

            ),

            opacity = 0.5,

        ))

layout = dict(

        title = 'Flights in brazil<br>(Click and drag to move and use wheel to zoom in)',

        showlegend = False,         

        geo = dict(

            resolution = 50,

            showland = True,

            showlakes = True,

            landcolor = 'rgb(204, 204, 204)',

            countrycolor = 'rgb(204, 204, 204)',

            lakecolor = 'rgb(255, 255, 255)',

            projection = dict( type="equirectangular" ),

            coastlinewidth = 2,

            lataxis = dict(

                range = [ -60, 60 ],

                showgrid = True,

                tickmode = "linear",

                dtick = 10

            ),

            lonaxis = dict(

                range = [-110, 100],

                showgrid = True,

                tickmode = "linear",

                dtick = 20

            ),

        )

    )

    

fig = dict( data=flight_paths + airports, layout=layout )

pyo.iplot( fig, validate=False)
df_time = df[['Flights', 'Airline', 'Flight_Type', 'Departure_Estimate',

       'Departure_Real', 'Arrival_Estimate', 'Arrival_Real',

       'Flight_Situation', 'Origin_Airport',

       'Origin_City', 'Origin_State', 'Origin_Country', 'Destination_Airport',

       'Destination_City', 'Destination_State', 'Destination_Country',

       'Destination_Long', 'Destination_Lat', 'Origin_Long', 'Origin_Lat']]

df_time.dropna(how='any',inplace=True)

df_time['Departure_Estimate'] = pd.to_datetime(df_time['Departure_Estimate'])

df_time['Departure_Real'] = pd.to_datetime(df_time['Departure_Real'])

df_time['Arrival_Estimate'] = pd.to_datetime(df_time['Arrival_Estimate'])

df_time['Arrival_Real'] = pd.to_datetime(df_time['Arrival_Real'])

df_time['Departure_Delays'] =df_time.Departure_Real - df_time.Departure_Estimate

df_time['Arrival_Delays'] = df_time.Arrival_Real - df_time.Arrival_Estimate

df_time['Departure_Delays'] = df_time['Departure_Delays'].apply(lambda x : round(x.total_seconds()/60))

df_time['Arrival_Delays'] = df_time['Arrival_Delays'].apply(lambda x : round(x.total_seconds()/60))
#__________________________________________________________________

# function that extract statistical parameters from a grouby objet:

def get_stats(group):

    return {'min': group.min(), 'max': group.max(),

            'count': group.count(), 'mean': group.mean()}

#_______________________________________________________________

# Creation of a dataframe with statitical infos on each airline:

global_stats = df_time['Departure_Delays'].groupby(df['Airline']).apply(get_stats).unstack()

global_stats = global_stats.sort_values(by = 'count', ascending=False)

global_stats.head(10)
f, axes = plt.subplots(2, 1, figsize=(10,24))

plt.sca(axes[0])

TopList = np.array(global_stats.head(20).index)

TopData = df_time[df_time['Airline'].isin(TopList)]

ax = sns.lvplot(data=TopData, y='Airline', x='Departure_Delays',order=TopList, hue="Flight_Type")

plt.title('The Departure Delays for Top 20 Airlines', fontsize = 18)

ax.yaxis.label.set_visible(False)

plt.sca(axes[1])

ax = sns.lvplot(data=TopData, y='Airline', x='Arrival_Delays',order=TopList, hue="Flight_Type")

plt.title('The Arrival Delays for Top 20 Airlines', fontsize = 18)

ax.yaxis.label.set_visible(False)

plt.show()
delay_type = lambda x:((0,1)[x > 10],2)[x > 60]

TopData['DELAY_LEVEL'] = TopData['Departure_Delays'].apply(delay_type)

fig = plt.figure(1, figsize=(10,10))

ax = sns.countplot(y="Airline", hue='DELAY_LEVEL', data = TopData)

# Set the legend

L = plt.legend()

L.get_texts()[0].set_text('on time (t < 10 min)')

L.get_texts()[1].set_text('small delay (10 < t < 60 min)')

L.get_texts()[2].set_text('large delay (t > 60 min)')

ax.yaxis.label.set_visible(False)

plt.title('The Departure Delays Types for top 20 Airlines', fontsize = 18)

plt.show()
airport_stats = df_time['Departure_Delays'].groupby(df['Origin_Airport']).apply(get_stats).unstack()

airport_stats = airport_stats.sort_values(by = 'count', ascending=False)

airport_stats.head(10)
f, axes = plt.subplots(2, 1, figsize=(10,24))

plt.sca(axes[0])

TopList = np.array(airport_stats.head(20).index)

TopData = df_time[df_time['Origin_Airport'].isin(TopList)]

ax = sns.lvplot(data=TopData, y='Origin_Airport', x='Departure_Delays',order=TopList, hue="Flight_Type")

plt.title('The Departure Delays for Top 20 Airports', fontsize = 18)

ax.yaxis.label.set_visible(False)

plt.sca(axes[1])

ax = sns.lvplot(data=TopData, y='Origin_Airport', x='Arrival_Delays',order=TopList, hue="Flight_Type")

plt.title('The Arrival Delays for Top 20 Airports', fontsize = 18)

ax.yaxis.label.set_visible(False)

plt.show()
delay_type = lambda x:((0,1)[x > 10],2)[x > 60]

TopData['DELAY_LEVEL'] = TopData['Departure_Delays'].apply(delay_type)

fig = plt.figure(1, figsize=(10,10))

ax = sns.countplot(y="Origin_Airport", hue='DELAY_LEVEL', data = TopData)

# Set the legend

L = plt.legend()

L.get_texts()[0].set_text('on time (t < 10 min)')

L.get_texts()[1].set_text('small delay (10 < t < 60 min)')

L.get_texts()[2].set_text('large delay (t > 60 min)')

ax.yaxis.label.set_visible(False)

plt.title('The Departure Delay Type for Top 20 Airports', fontsize = 18)

plt.show()