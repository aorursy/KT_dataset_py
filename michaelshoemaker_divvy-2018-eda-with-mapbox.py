#So I can see what's in the directories
import os
#If I want to know how long it takes to complete a block of code
import time
#To move my mapbox token function from the input to the working directory
from shutil import copyfile
#For cool map visuals with mapbox
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
#Because I like it when pandas works
import numpy as np 
#Well, it's Kaggle and I'm using Python so yeah, pandas.
import pandas as pd 
#Easy count plots heatmaps etc.
import seaborn as sns
#Never hurts to have this one and I'm using seaborn
import matplotlib.pyplot as plt
%matplotlib inline
#Move the python file with my function which returns my token
copyfile(src = "../input/mboxload/mapbox.py", dst = "../working/private_mapbox_access_token.py")
#Get the function I titled "mapboxtoken"
from private_mapbox_access_token import mapboxtoken
#Load the token string into a variable called token
token = mapboxtoken()
#Make sure all of my files are where they should be
!ls '../input'
!ls '../working'
start_time = time.time()
df1 = pd.read_csv(r'../input/divvy-bike-chicago-2018/Divvy_Trips_2018_Q1.csv',parse_dates=['start_time','end_time'])
df2 = pd.read_csv(r'../input/divvy-bike-chicago-2018/Divvy_Trips_2018_Q2.csv',parse_dates=['start_time','end_time'])
df3 = pd.read_csv(r'../input/divvy-bike-chicago-2018/Divvy_Trips_2018_Q3.csv',parse_dates=['start_time','end_time'])
df4 = pd.read_csv(r'../input/divvy-bike-chicago-2018/Divvy_Trips_2018_Q4.csv',parse_dates=['start_time','end_time'])
frames = [df1, df2, df3, df4]
df = pd.concat(frames)
print("--- {} seconds ---".format(time.time() - start_time))
df.sample(3)
stations = pd.read_csv(r'../input/bikestations/Divvy_Bicycle_Stations.csv')
stations.sample(3)
df[df['from_station_id']==451].head(1)
print('There are {} ID numbers in the stations dataframe'
      '\nand {} in the df dataframe'
      .format(stations['ID'].nunique(), df['from_station_id'].nunique()))
combined =  pd.merge(df, stations,how='inner', left_on = 'from_station_id', right_on = 'ID')
plt.figure(figsize=(12,8))
sns.heatmap(combined.isnull(),cmap='viridis',yticklabels=False)
combined[combined['gender'].isnull()|combined['birthyear'].isnull()].sample(8)
sns.countplot(x='usertype', data = combined)
cust = len(combined[combined['usertype']=='Customer'])
subs = len(combined[combined['usertype']=='Subscriber'])
tot = len(combined) 
print('Users of Divvy in 2018 were {}% Customers and {}% Subscribers'.format(round((cust/tot)*100,2)
                                                                             ,round((subs/tot)*100,2)))
sns.countplot(x='usertype', hue = 'gender', data = combined)
combined['start_date'] = combined['start_time'].dt.date
datecount = pd.DataFrame(combined['trip_id'].groupby(combined['start_date']).count())
plt.figure(figsize=(20,12))
plt.plot(datecount)
plt.xlabel("Date")
plt.ylabel("Rides")
plt.figure(figsize=(20,12))
plt.plot(datecount.rolling(7, min_periods=1).mean(),color = 'orange')
plt.xlabel("Date")
plt.ylabel("Rides")
fig, ax = plt.subplots(figsize=(15,7))
combined.groupby(['start_date','gender']).count()['trip_id'].unstack().rolling(7, min_periods=1).mean().plot(ax=ax)
fig, ax = plt.subplots(figsize=(15,7))
combined.groupby(['start_date','usertype']).count()['trip_id'].unstack().rolling(7, min_periods=1).mean().plot(ax=ax)
combined['dayofweek'] = combined['start_time'].dt.day_name()
daycount = pd.DataFrame(combined.groupby('dayofweek').agg('count')['trip_id'])
daycount.reset_index(inplace=True)
daycount.columns = ['dayofweek','trips']
order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
daycount = daycount.groupby(['dayofweek']).sum().reindex(order) 
daycount.reset_index(inplace = True)
day = daycount['dayofweek']
cnt = daycount['trips']
data = [go.Bar(x=day, y = cnt)]

layout = go.Layout(title = "Sum of all trips for each day of week")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
freq = combined.groupby(['Latitude', 'Longitude','from_station_name']).agg('count')['trip_id'].sort_values(ascending=False).reset_index()
freq.columns = ['lat','lon','station','count']
top50 = freq[freq.index<50]
top50
ls = ['Addr: {} \n Uses in 2018:{}'.format(top50['station'].iloc[i],top50['count'].iloc[i]) for i in range(len(top50))]
data = [go.Scattermapbox(
            lat= top50['lat'] ,
            lon= top50['lon'],
            mode='markers',
            text = ls,
            hoverinfo = 'text',
            marker=dict(
                #Why divided by 3000 here? Because if I didn't the red would fill up the screen and I just played
                #around with this number until it looked pretty. 
                size= top50['count']/3000,
                color = 'red',
                opacity = .8,
            ),
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken=token,
                                bearing=10,
                                pitch=60,
                                zoom=12,
                                center= dict(
                                         lat=41.895278,
                                         lon=-87.636820),
                                #style= "mapbox://styles/shaz13/cjiog1iqa1vkd2soeu5eocy4i"
                               ),
                    width=900,
                    height=600, title = "Top 50 used stations in 2018")
fig = dict(data=data, layout=layout)
iplot(fig)
data = [go.Scattermapbox(
            lat= freq['lat'] ,
            lon= freq['lon'],
            customdata = freq['station'],
            mode='markers',
            marker=dict(
                size= 4,
                color = 'red',
                opacity = .8,
            ),
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken=token,
                                bearing=10,
                                pitch=60,
                                zoom=10,
                                center= dict(
                                         lat=41.881832,
                                         lon=-87.623177),
                                #style= "mapbox://styles/shaz13/cjiog1iqa1vkd2soeu5eocy4i"
                               ),
                    width=900,
                    height=600, title = "Divvy Racks in 2018")
fig = dict(data=data, layout=layout)
iplot(fig)