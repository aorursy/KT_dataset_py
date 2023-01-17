import pandas as pd
import numpy as np
import os 
import plotly.offline as pyoff
from plotly.offline import iplot,plot,init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode()
import matplotlib.pyplot as plt
bike_share = pd.read_csv('../input/data.csv')
bike_share.columns
bike_share.dtypes
bike_share.head()
cat_cols = ['year','day','week']
bike_share.loc[:,cat_cols] = bike_share.loc[:,cat_cols].astype(str) # converting the categorcical columns to str type 
trips_year = pd.DataFrame(bike_share.year.value_counts()).reset_index()
trips_year.columns = ['year','trips']
trips_year = trips_year.sort_values('year')
# trips_year.to_csv("trips_year.csv",index = False)
# trips_year = pd.read_csv("trips_year.csv")
ty = go.Bar(x = "Year-"+trips_year.year.astype(str) ,y = trips_year.trips,name = 'Trips',
           text = trips_year.trips,textposition = 'auto')
layout = go.Layout(title = 'Number of trips taken every year')

data = [ty]
fig = go.Figure(data= data,layout =layout)
iplot(fig)
from datetime import datetime
dates = bike_share.starttime.apply(lambda x : datetime.strptime(x,"%Y-%m-%d %H:%M:%S").date())
bike_share['startdateonly'] = dates
daily = bike_share.groupby(['startdateonly']).size().to_frame().reset_index()
daily.columns = ['date','trips']
# daily.to_csv("daily.csv",index = False)
# daily= pd.read_csv("daily.csv")
data = [go.Scatter(
          x=daily.date,
          y=daily.trips)]
layout = layout = go.Layout(title = "Rides taken over the years (timeline)",
    xaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    )
)
fig = go.Figure(data=data,layout = layout)
iplot(fig)
###### Gender distribution plot
gender_dist = go.Bar( x= bike_share.gender.value_counts().index,
                     y = bike_share.gender.value_counts(),
                     text = bike_share.gender.value_counts(),
                     textposition = 'auto'
                    )
layout = go.Layout(title = 'Gender Distribution in the dataset')
data = [gender_dist]
fig = go.Figure(data=data,layout = layout)
iplot(fig)
daily_gender = bike_share.groupby(['startdateonly','gender']).size().to_frame().reset_index()
daily_gender.columns = ['startdateonly','gender','trips']
# daily_gender.to_csv('daily_gender.csv',index=False)
# daily_gender = pd.read_csv('daily_gender.csv')
trace_Male = go.Scatter(
                x=daily_gender.startdateonly[daily_gender.gender =='Male'],
                y=daily_gender.trips[daily_gender.gender =='Male'],
                name = "Male",
                line = dict(color = 'blue'),
                opacity = 0.8)

trace_Female = go.Scatter(
                x=daily_gender.startdateonly[daily_gender.gender =='Female'],
                y=daily_gender.trips[daily_gender.gender =='Female'],
                name = "Female",
                line = dict(color = 'green'),
                opacity = 0.8)
layout = layout = go.Layout(title = "Rides taken over the years based on gender",
    xaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    )
)
data = [trace_Male,trace_Female]
fig = go.Figure(data=data,layout = layout)
iplot(fig)
rides_hour = bike_share.groupby(['hour']).size().reset_index()
rides_hour.columns = ['hour','trips']
# rides_hour.to_csv('rides_hour.csv',index= False)
# rides_hour = pd.read_csv('rides_hour.csv')
rides_hour.head()
hour = go.Bar(x = rides_hour.hour.astype(str),
             y = rides_hour.trips, text = rides_hour.trips,textposition = 'auto')
layout = go.Layout(title = 'Rides taken at different hour of the day')
data = [hour]
fig = go.Figure(data= data,layout=layout)
iplot(fig)
morning_rides_start_location = bike_share.to_station_name[(bike_share.hour > 8) & bike_share.hour<9].value_counts()
evening_rides_end_location = bike_share.from_station_name[(bike_share.hour > 16)& (bike_share.hour <19)].value_counts() 
len([i for i in morning_rides_start_location.index if i in evening_rides_end_location.index])
filtered_time_rides_morning = bike_share[(8<bike_share.hour) & (bike_share.hour>9)]
filtered_time_rides_evening = bike_share[(17<bike_share.hour) & (bike_share.hour>19)]
filtered_time_rides_morning_grouped = filtered_time_rides_morning.groupby(['to_station_name']).size().to_frame('trips').reset_index()
filtered_time_rides_morning_grouped = filtered_time_rides_morning.groupby(['to_station_name']).size().to_frame('trips').reset_index()
filtered_time_rides_evening_grouped = filtered_time_rides_evening.groupby(['from_station_name']).size().to_frame('trips').reset_index()
filtered_time_rides_morning_grouped.columns = ['station','trips_to']
filtered_time_rides_evening_grouped.columns = ['station','trips_from']
rides_from_to_df = filtered_time_rides_morning_grouped.merge(filtered_time_rides_evening_grouped)
rides_from_to_df = rides_from_to_df.sort_values(['trips_to','trips_from'],ascending=False)
# rides_from_to_df.to_csv('rides_from_to_df.csv',index = False)
# rides_from_to_df = pd.read_csv('rides_from_to_df.csv')
rides_from_to_df.head()
rides_from_to_df_plot = rides_from_to_df.head(10)
trips_to = go.Bar(x = rides_from_to_df_plot.station,
                  y = rides_from_to_df_plot.trips_to,
                  text = rides_from_to_df_plot.trips_to,
                  textposition = 'auto',name = 'Morning trips ended in')
trips_from = go.Bar(x = rides_from_to_df_plot.station,
                  y = rides_from_to_df_plot.trips_from,
                  text = rides_from_to_df_plot.trips_from,
                  textposition = 'auto',name = 'Evening trips started from')
data = [trips_to,trips_from]
layout = go.Layout(title ='Trips taken in the morning and trips taken in the evening<br>(Top 10 based on counts of places where the trips ended)')
fig = go.Figure(data=data,layout = layout)
iplot(fig)
daily_usertype = bike_share.groupby(['startdateonly','usertype']).size().to_frame().reset_index()
daily_usertype.columns = ['startdateonly','usertype','trips']
# daily_usertype.to_csv('daily_usertype.csv',index = False)
# daily_usertype = pd.read_csv('daily_usertype.csv')
daily_usertype.usertype.value_counts()
daily_usertype.head()
trace_Subscriber = go.Scatter(
                x=daily_usertype.startdateonly[daily_usertype.usertype =='Subscriber'],
                y=daily_usertype.trips[daily_usertype.usertype =='Subscriber'],
                name = "Subscriber",
                opacity = 0.8)

trace_Customer = go.Scatter(
                x=daily_usertype.startdateonly[daily_usertype.usertype =='Customer'],
                y=daily_usertype.trips[daily_usertype.usertype =='Customer'],
                name = "Customer",
                opacity = 0.8)
trace_Dependent = go.Scatter(
                x=daily_usertype.startdateonly[daily_usertype.usertype =='Dependent'],
                y=daily_usertype.trips[daily_usertype.usertype =='Dependent'],
                name = "Dependent",
                opacity = 0.8)


layout = layout = go.Layout(title = "Rides taken over the years based on customer type",
    xaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    )
)
data = [trace_Subscriber,trace_Customer,trace_Dependent]
fig = go.Figure(data=data,layout = layout)
iplot(fig)
bike_share.starttime[0] ,bike_share.stoptime[0] , bike_share.tripduration[0]
##### Are people riding more over the years?
trip_duration_median = bike_share.groupby(['startdateonly']).agg({'tripduration':np.median}).reset_index()
# trip_duration_median.to_csv('trip_duration_median.csv',index = False)
# trip_duration_median =pd.read_csv('trip_duration_median.csv')
trip_duration_median.columns
data = [go.Scatter(
          x=trip_duration_median.startdateonly,
          y=trip_duration_median.tripduration)]
layout = layout = go.Layout(title = "Median ride time over the years",
    xaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    )
)
fig = go.Figure(data=data,layout = layout)
iplot(fig)
trip_duration_agg_df = bike_share.groupby(['startdateonly']).agg({'tripduration':{
                                                                        'medianduration' : np.median,
                                                                        'minduration' : np.min,
                                                                        'maxduration' : np.max,
                                                                }}).reset_index()
trip_duration_agg_df.columns  = ['startdateonly', 'medianduration', 'minduration', 'maxduration']
# trip_duration_agg_df.to_csv('trip_duration_agg_df.csv',index = False)
# trip_duration_agg_df = pd.read_csv('trip_duration_agg_df.csv')
trip_duration_agg_df.head()
minduration = go.Scatter(
          x=trip_duration_agg_df.startdateonly,
          y=trip_duration_agg_df.minduration,name ='Minimum duration' )
               
maxduration = go.Scatter(
          x=trip_duration_agg_df.startdateonly,
          y=trip_duration_agg_df.maxduration,name = 'Maximum duration')

medianduration = go.Scatter(
          x=trip_duration_agg_df.startdateonly,
          y=trip_duration_agg_df.medianduration,name = 'Median duration')

data = [minduration,medianduration,maxduration]

layout = layout = go.Layout(title = "Ride time over the years",
    xaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    )
)
fig = go.Figure(data=data,layout = layout)
iplot(fig)
usertype_ridetime_df = bike_share.groupby('usertype').agg({'tripduration' : {'minduration':np.min,
                                                     'medianduration':np.median,
                                                     'maxduration':np.max,
                                                    }}).reset_index()
usertype_ridetime_df.columns = ['usertype','minduration','medianduration','maxduration']
# usertype_ridetime_df.to_csv('usertype_ridetime_df.csv',index = False)
# usertype_ridetime_df = pd.read_csv("usertype_ridetime_df.csv")
usertype_ridetime_df.head()
minduration = go.Bar(
          x=usertype_ridetime_df.usertype, 
          y=usertype_ridetime_df.minduration.round(2),
    text = usertype_ridetime_df.minduration.round(2),textposition = 'auto' ,
    name ='Minimum duration' )
               
maxduration = go.Bar(
          x=usertype_ridetime_df.usertype,
          y=usertype_ridetime_df.maxduration.round(2),
     text = usertype_ridetime_df.maxduration.round(2),textposition = 'auto' ,
    name = 'Maximum duration')

medianduration = go.Bar(
          x=usertype_ridetime_df.usertype,
          y=usertype_ridetime_df.medianduration.round(2),
     text = usertype_ridetime_df.medianduration.round(2),textposition = 'auto' ,
    name = 'Median duration')

data = [minduration,medianduration,maxduration]

layout = layout = go.Layout(title = "Median Ride time of different types of customers",
    xaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    )
)
fig = go.Figure(data=data,layout = layout)
iplot(fig)
usertype_events_counts_df = bike_share.groupby('events').size().to_frame('trips').reset_index()
usertype_events_counts_df
# usertype_events_counts_df.to_csv('usertype_events_counts_df.csv',index=False)
# usertype_events_counts_df = pd.read_csv('usertype_events_counts_df.csv')
trace1 = go.Bar(x = usertype_events_counts_df.events,
               y=usertype_events_counts_df.trips,
                name = 'Number of trips')
layout = go.Layout(title = 'Number of trips in different weather conditions')

data = [trace1]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
usertype_events_counts_years_df = bike_share.groupby(['events','year']).size().to_frame('trips').reset_index()
# usertype_events_counts_years_df.to_csv('usertype_events_counts_years_df.csv',index=False)
# usertype_events_counts_years_df = pd.read_csv('usertype_events_counts_years_df.csv')
usertype_events_counts_years_df.head()
x = usertype_events_counts_years_df.year.unique()
clear = usertype_events_counts_years_df[usertype_events_counts_years_df.events == 'clear'].trips
cloudy = usertype_events_counts_years_df[usertype_events_counts_years_df.events == 'cloudy'].trips
not_clear = usertype_events_counts_years_df[usertype_events_counts_years_df.events == 'not clear'].trips
rain_or_snow = usertype_events_counts_years_df[usertype_events_counts_years_df.events == 'rain or snow'].trips
tstorms = usertype_events_counts_years_df[usertype_events_counts_years_df.events == 'tstorms'].trips
unknown = usertype_events_counts_years_df[usertype_events_counts_years_df.events == 'unknown'].trips

clear  = go.Bar(x =x,
               y= clear,name = 'clear')
cloudy  = go.Bar(x =x,
               y= cloudy,name = 'cloudy')

not_clear  = go.Bar(x =x,
               y= not_clear,name = 'not_clear')

rain_or_snow  = go.Bar(x =x,
               y= rain_or_snow,name = 'rain_or_snow')

tstorms  = go.Bar(x =x,
               y= tstorms,name = 'tstorms')

unknown  = go.Bar(x =x,
               y= unknown,name = 'unknown')

layout = go.Layout(title = 'Number of rides in different weather conditions over the years')
data = [clear,cloudy,not_clear,rain_or_snow,tstorms,unknown]

fig = go.Figure(data= data,layout= layout)
iplot(fig)
x = usertype_events_counts_years_df.events.unique()
usertype_events_counts_years_df.year = usertype_events_counts_years_df.year.astype(str)
y2014 = usertype_events_counts_years_df[usertype_events_counts_years_df.year == '2014'].trips
y2015 = usertype_events_counts_years_df[usertype_events_counts_years_df.year == '2015'].trips
y2016 = usertype_events_counts_years_df[usertype_events_counts_years_df.year == '2016'].trips
y2017 = usertype_events_counts_years_df[usertype_events_counts_years_df.year == '2017'].trips
y2014  = go.Bar(x =x,
               y= y2014,name = '2014')
y2015  = go.Bar(x =x,
               y= y2015,name = '2015')

y2016  = go.Bar(x =x,
               y= y2016,name = '2016')

y2017  = go.Bar(x =x,
               y= y2017,name = '2017')

layout = go.Layout(title = 'Distribution of rides in different weather conditions over the years')
data = [y2014,y2015,y2016,y2017]

fig = go.Figure(data= data,layout= layout)
iplot(fig)
bike_share.columns
print("There are {} unique stations in the dataset".format(len(bike_share.from_station_name.unique())))
len(bike_share.from_station_name.value_counts().index[bike_share.from_station_name.value_counts()>10000])
len(bike_share.from_station_name.value_counts()>10000)
rides_stations_weather = bike_share.groupby(['from_station_name','events']).size().to_frame().reset_index()
rides_stations_weather.columns = ['from_station_name','events','trips']
rides_stations_weather = rides_stations_weather.sort_values(['from_station_name','events'])
# rides_stations_weather.to_csv('rides_stations_weather.csv',index=False)
# rides_stations_weather = pd.read_csv('rides_stations_weather.csv')
data = []
for i in rides_stations_weather.events.unique():
    data.append(go.Bar (x = rides_stations_weather[rides_stations_weather.events == i].from_station_name,
                       y = rides_stations_weather[rides_stations_weather.events == i].trips,
                       name = i))
layout = go.Layout(title = 'Weather conditions at different stations',barmode='stack')
fig = go.Figure(data= data,layout = layout)
iplot(fig)
usertype_events_df = bike_share.groupby('events').agg({'tripduration' : {'minduration':np.min,
                                                     'medianduration':np.median,
                                                     'maxduration':np.max,
                                                    }}).reset_index()
usertype_events_df.columns = ['events','minduration','medianduration','maxduration']
# usertype_events_df.to_csv('usertype_events_df.csv',index=False)
# usertype_events_df = pd.read_csv('usertype_events_df.csv') 
minduration = go.Bar(
          x=usertype_events_df.events,
          y=usertype_events_df.minduration.round(2),
    text = usertype_events_df.minduration.round(2),textposition = 'auto',
    name ='Minimum duration' )
               
maxduration = go.Bar(
          x=usertype_events_df.events,
          y=usertype_events_df.maxduration.round(2),
    text = usertype_events_df.maxduration.round(2),textposition = 'auto',
    name = 'Maximum duration')

medianduration = go.Bar(
          x=usertype_events_df.events,
          y=usertype_events_df.medianduration.round(2),
     text = usertype_events_df.medianduration.round(2),textposition = 'auto',
    name = 'Median duration')

data = [minduration,medianduration,maxduration]

layout = layout = go.Layout(title = "Ride time in different weather conditions",
    xaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    ),
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
#         showline=True,
#         ticks='',
#         showticklabels=False
    )
)
fig = go.Figure(data=data,layout = layout)
iplot(fig)
rides_customertype = bike_share.groupby(['events','usertype']).size().to_frame('trips').reset_index()
rides_customertype = rides_customertype.pivot_table(index= ['events'],columns= ['usertype'],aggfunc=[np.sum],fill_value=0).reset_index()
rides_customertype.columns = ['events','Customer','Dependent','Subscriber']
rides_customertype['Other'] = rides_customertype.Dependent+rides_customertype.Customer
# Here I have divided each value in the Customer and Other column by the corresponsing column total and multiplied it by 100
rides_customertype['OtherPer'] = (rides_customertype.Other/rides_customertype.Other.sum())*100
rides_customertype['SubscriberPer'] = (rides_customertype.Subscriber/rides_customertype.Subscriber.sum())*100
# rides_customertype.to_csv('rides_customertype.csv',index=False)
# rides_customertype = pd.read_csv('rides_customertype.csv')
rides_customertype
OtherPer = go.Bar(x = rides_customertype.events,
                 y = rides_customertype.OtherPer,
                name = "Other")
SubscriberPer = go.Bar(x = rides_customertype.events,
                 y = rides_customertype.SubscriberPer,
                       name = "Subscriber")

layout = go.Layout(title = 'Comparing the Number of rides taken by different types of customers <br> in different weather coditions')

data = [OtherPer,SubscriberPer]

fig = go.Figure(data= data, layout=layout)

iplot(fig)
bike_share.columns
lat_lon = bike_share.groupby(['from_station_name']).agg({'latitude_start': {'lat': np.median},
                                                         'longitude_start':{'lon':np.median}}).reset_index()
lat_lon.columns = ['from_station_name','lat','lon']
airports = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = lat_lon['lon'],
        lat = lat_lon['lat'],
        hoverinfo = 'text',
#         text = df_airports['airport'],
        mode = 'markers',
        marker = dict( 
            size=2, 
            color='rgb(255, 0, 0)',
            line = dict(
                width=3,
                color='rgba(68, 68, 68, 0)'
            )
        ))]
flight_paths = []
for i in range( len( bike_share.trip_id[0:1000] ) ):
    flight_paths.append(
        dict(
            type = 'scattergeo',
            locationmode = 'USA-states',
            lon = [ bike_share['longitude_start'][i], bike_share['longitude_end'][i] ],
            lat = [ bike_share['latitude_start'][i], bike_share['latitude_end'][i] ],
            mode = 'lines',
            line = dict(
                width = 1,
                color = 'red',
            ),
#             opacity = 0.3,
        )
    )
layout = dict(
        title = 'Rides taken in chicago over all the years<br>(Hover for Station names) <br> (plotting only first 1000 rides for now)',
        showlegend = False, 
        geo = dict(
            scope='north america',
            projection=dict(type='azimuthal equal area'),
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
        ),
    ) 
fig = dict( data=flight_paths + airports, layout=layout )
iplot(fig)