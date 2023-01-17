
# -*- coding: utf-8 -*-

import plotly.offline as py
import plotly.graph_objs as go

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.cm as cmx

from sklearn.linear_model import LinearRegression

py.init_notebook_mode(connected=True)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


#Data
flights0 = pd.read_csv("../input/flights-sample-2/susa.csv", sep=",", dtype={'IATA_CODE':str, 'TAXI_IN':float})
flights0 = flights0.iloc[:,1:32]
flights0.shape
flights0['ROUTE'] = tuple(zip(flights0['ORIGIN_AIRPORT'], flights0['DESTINATION_AIRPORT']))
airports = pd.read_csv("../input/flight-delays/airports.csv", sep=",", dtype={'LONGITUDE':float, 'LATITUDE':float})

#Estimate flight delay
X = flights0.DISTANCE.values.reshape(-1,1)
y = flights0.SCHEDULED_TIME.values.reshape(-1,1)
lm = LinearRegression()
lm.fit(X, y)
flights0['FLIGHT_DELAY'] = np.around((np.array(flights0.AIR_TIME) - lm.coef_*np.array(flights0.DISTANCE)).T)

# Select unique routes and add the count of flights

# Here we can filter routes by Airline!
#flights0 = flights0.loc[flights0.AIRLINE=="DL"]

flights = flights0.copy()
flights = flights.drop_duplicates(subset='ROUTE')
flights = flights.sort_values(['ROUTE'])
flights['COUNT'] = flights0.groupby(['ROUTE']).size().values
flights['FLIGHT_DELAY'] = flights0.groupby(['ROUTE']).sum().FLIGHT_DELAY.values/flights.COUNT

maxcount_f = max(flights['COUNT'])


# Pick the routes with most flights
flights = flights.loc[flights['COUNT']>flights.COUNT.quantile(q=0.9)]
flights = flights.reset_index()

# Select the airports that are in flights
airports = airports.loc[(airports.IATA_CODE.isin(flights.ORIGIN_AIRPORT)) | (airports.IATA_CODE.isin(flights.DESTINATION_AIRPORT))]
airports = airports.reset_index()

# Calculate departure_delay, taxi_out y taxi_in medios y count
## Stats out
### Calculate sum departure_delay and taxi_out by origin airport
stats = flights0.loc[flights0.ORIGIN_AIRPORT.isin(airports.IATA_CODE),:].groupby(['ORIGIN_AIRPORT']).sum().DEPARTURE_DELAY.reset_index()
taxi_out = flights0.loc[flights0.ORIGIN_AIRPORT.isin(airports.IATA_CODE),:].groupby(['ORIGIN_AIRPORT']).sum().TAXI_OUT.reset_index()
### Calculate amount of flights by origin airport
out_flights = flights0.loc[flights0.ORIGIN_AIRPORT.isin(airports.IATA_CODE),:].groupby(['ORIGIN_AIRPORT']).size().reset_index()
### Calculate mean departure delay by airport 
stats['DEPARTURE_DELAY'] = stats.DEPARTURE_DELAY/out_flights[0]
### Calculate mean taxi out
stats['TAXI_OUT'] = taxi_out.TAXI_OUT/out_flights[0]
airports = airports.merge(stats, left_on='IATA_CODE', right_on='ORIGIN_AIRPORT', how='outer')
## Stats in
### Calculate sum tax in by destination airport
stats = flights0.loc[flights0.DESTINATION_AIRPORT.isin(airports.IATA_CODE),:].groupby(['DESTINATION_AIRPORT']).sum().TAXI_IN.reset_index()
in_flights = flights0.loc[flights0.DESTINATION_AIRPORT.isin(airports.IATA_CODE),:].groupby(['DESTINATION_AIRPORT']).size().reset_index()
stats['TAXI_IN'] = stats.TAXI_IN/in_flights[0]
airports = airports.merge(stats, left_on='IATA_CODE', right_on='DESTINATION_AIRPORT', how='outer')
airports['COUNT'] = in_flights[0] + out_flights[0]
airports = airports.drop(['index','ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'], axis=1)

#SCORE
airports['SCORE'] = airports['DEPARTURE_DELAY']+airports['TAXI_OUT']+airports['TAXI_IN']

# Codification of delays to colors
cmap = plt.cm.autumn_r
norm = mc.Normalize(-24.887323943661972,27.181818181818183)# min and max of flights_delay from the whole flights
color = cmx.ScalarMappable(cmap = cmap, norm=norm).to_rgba(flights.FLIGHT_DELAY, bytes = True)
color = ['rgba(' + str(x[0]) + ', ' + str(x[1]) + ', ' + str(x[2]) + ', ' + str(x[3]) + ')' for x in color]

# Group routes. AB==BA            
flights.ROUTE = tuple(map(sorted, flights.ROUTE))
flights.ROUTE = list(map(tuple, flights.ROUTE))
f = {'COUNT': 'sum', 'FLIGHT_DELAY': 'mean'}
flights = flights.groupby('ROUTE').agg(f).reset_index()


airportspy = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = airports['LONGITUDE'],
        lat = airports['LATITUDE'],
        text = '<b>' + airports['AIRPORT'].map(str) + '</b>' \
        + '<br>' + 'Score: ' + round(airports['SCORE'].astype(np.double),1).map(str) + ' minutes' \
        + '<br>' + 'Departure delay: ' + round(airports['DEPARTURE_DELAY'].astype(np.double),1).map(str) + ' minutes' \
        + '<br>' + 'Taxi out: ' + round(airports['TAXI_OUT'].astype(np.double),1).map(str) + ' minutes' \
        + '<br>' + 'Taxi in: ' + round(airports['TAXI_IN'].astype(np.double),1).map(str) + ' minutes',
        hoverinfo = 'text',
        hoverlabel = dict(
              bordercolor = np.array(airports.SCORE)            
        ),
        marker = dict(
            size=np.array(airports.COUNT*20/max(airports.COUNT)),
            sizemin=2,
            color = np.array(airports.SCORE),
            cmin = 10.565915336317195,
            cmax = 38.383341806128094,
            colorscale = [[0,'rgb(255, 255, 0)'],[1,'rgb(255, 0, 0)']],
            colorbar=go.ColorBar(
                 title='Score'
            ),
            opacity = 1,
            line = dict(
                width=0,
                color='rgba(68, 68, 68, 0)'
            )          
        ))]


flight_paths = []
for i in range(len(flights)):
      
    aux_origin = airports.loc[airports['IATA_CODE']==list(zip(*flights.ROUTE))[0][i]]
    aux_destin = airports.loc[airports['IATA_CODE']==list(zip(*flights.ROUTE))[1][i]]  
    
    flight_paths.append(
        dict(
            type = 'scattergeo',
            locationmode = 'USA-states',
            lon = [ float(aux_origin['LONGITUDE']), float(aux_destin['LONGITUDE']) ],
            lat = [ float(aux_origin['LATITUDE']), float(aux_destin['LATITUDE']) ],
            mode = 'lines',
            hoverinfo = 'skip',
            line = dict(
                width = 0.5,
                color = color[i]
            ),
            opacity = float(flights.loc[i,'COUNT']/maxcount_f)*2
        )
    )
    
layout = go.Layout(
        showlegend = False, 
        geo = dict(
            scope='north america',
            projection=dict( type='orthographic' , scale = 1.8),
            showland = True,
            showocean = True,
            showcoastlines = True,
            showcountries = True,
            landcolor = 'rgb(49, 49, 49)',
            countrycolor = 'rgb(90, 90, 90)',
            coastlinecolor = 'rgb(90, 90, 90)',
            oceancolor = 'rgb(29, 29, 29)',
            bgcolor = 'rgb(29, 29, 29)',
            center = dict(lon='-100', lat='36')
        ),
        margin=go.Margin(
         l=0,
         r=0,
         b=0,
         t=0,
         pad=0
        ),
        autosize=True,
        paper_bgcolor = 'rgb(29, 29, 29)',
        plot_bgcolor = 'rgb(29, 29, 29)'
                            )

fig = dict( data=flight_paths + airportspy, layout=layout )
py.iplot(fig, filename='routes-graph')