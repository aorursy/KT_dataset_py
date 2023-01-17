#Imports

import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt



%matplotlib inline



import plotly.plotly as ply

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



import warnings

warnings.filterwarnings('ignore')
#Get the data filtered to last three years

dfa = pd.read_csv('../input/nypd-motor-vehicle-collisions.csv')



dfa['DATE'] = pd.to_datetime(dfa['DATE'])



currentyear = dt.datetime.now().year

collisionyear = pd.DatetimeIndex(dfa['DATE']).year



dfa = dfa[pd.DatetimeIndex(dfa['DATE']).year > (currentyear - 4)]
#Create dataframe of fields of interest

dfa2 = dfa[['DATE','TIME','BOROUGH','ZIP CODE', 

            'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS KILLED', 'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST KILLED',

           'NUMBER OF PERSONS INJURED', 'NUMBER OF PEDESTRIANS INJURED','NUMBER OF CYCLIST INJURED', 'NUMBER OF MOTORIST INJURED',

           'CONTRIBUTING FACTOR VEHICLE 1','CONTRIBUTING FACTOR VEHICLE 2','LATITUDE','LONGITUDE']]



dfa2.columns = ['date', 'time','borough','zipcode','persons_killed','pedestrians_killed', 'cyclists_killed', 'motorists_killed',

                'persons_injured','pedestrians_injured', 'cyclists_injured', 'motorists_injured', 'vehicle1_reason',

                'vehicle2_reason','latitude','longitude']



dfa2.head(1)
#Consolidate Injured and Killed

dfa3 = dfa2[['date', 'time','borough','zipcode','vehicle1_reason','vehicle2_reason','latitude','longitude']]



dfa3['killed'] = dfa2.persons_killed + dfa2.pedestrians_killed + dfa2.cyclists_killed + dfa2.motorists_killed 

dfa3['injured'] = dfa2.persons_injured + dfa2.pedestrians_injured + dfa2.cyclists_injured + dfa2.motorists_injured 



dfa3.head(1)
# Percentage of missing values in each column

df_missing = pd.DataFrame(dfa3.isnull().sum(), columns = ['Count Missing Values'])

df_missing['% Missing Values'] = dfa3.isnull().sum()/len(dfa3)



df_missing
#When the borough is blank, replace with the text, "unknown"

listb = []



for i,v in enumerate(dfa3.borough):

    if pd.isnull(v):

        listb.append('UNKNOWN')

    else:

        listb.append(v)



dfa3['borough'] = listb
#When the zipcode is blank, replace with the text, 99999

listb = []



for i,v in enumerate(dfa3.zipcode):

    if pd.isnull(v):

        listb.append(99999)

    else:

        listb.append(v)



dfa3['zipcode'] = listb



dfa3.head(1)
df_ah = dfa3[['date','time']]

df_ah = df_ah[df_ah.date.dt.year == currentyear]



df_ah['time'] = pd.to_datetime(df_ah['time'])



df_ah['hour'] = df_ah.time.dt.hour



df_hourcount = df_ah.groupby('hour').count()



ax = df_hourcount.plot(kind='bar', color='blue',figsize=(15,5),rot=0)

ax.set_ylabel('# of accidents', fontsize=12)

ax.set_title('# of Accidents per Hour this Year', fontsize=12)

                          

plt.show()
df_apdw = pd.DataFrame(dfa3.date)

#df_apdw = df_apdw[pd.DatetimeIndex(df_apdw.date).year == currentyear]

df_apdw = df_apdw[df_apdw.date.dt.year == currentyear]



df_apdw['weekday'] = df_apdw.date.dt.dayofweek



df_weekdaycount = df_apdw.groupby('weekday').count()



ax = df_weekdaycount.plot(kind='bar', color='green',figsize=(8,5),rot=0)

ax.set_ylabel('# of accidents', fontsize=12)

ax.set_title('# of Accidents this Year by Day of Week')

                          

plt.show()
dfrc = dfa3[['date','vehicle1_reason']]

dfrc = dfrc[dfrc.date.dt.year == currentyear]



dfrc_chart = dfrc.groupby('vehicle1_reason').count()

dfrc_chart.columns = ['count']



dfrc_chart = dfrc_chart.sort_values('count',ascending=False)



dfrc_chart_top10 = dfrc_chart.head(10)



ax = dfrc_chart_top10.plot(kind='bar',color='red',figsize=(8,5),rot=90)

ax.set_xlabel('Collision Reason', fontsize=12)

ax.set_ylabel('Occurrences', fontsize=12)

ax.set_title('Top 10 Reasons for Collisions this Year',fontsize=12)

plt.show()
#Get the data

df_im = dfa3[['date','injured']]



df_im['year'] = df_im.date.dt.year

df_im['month'] = df_im.date.dt.month



df_im = df_im.drop(['date'], axis=1)



df_im.head(1)
#Pivot the dataframe to get each year as a column

df_im2 = df_im.pivot_table(index='month', columns='year', values='injured', aggfunc='count')



df_im2.columns = ['threeyearsago','twoyearsago','lastyear','thisyear']
#Plot the collisions per month for each year

plt.figure(figsize=(15,5))

plt.title('Collisions per Month in Past Three Years')

plt.xlabel('Month', fontsize=12)

plt.ylabel('# of Collisions', fontsize=12)

           

ax1 = df_im2.threeyearsago.plot(color='blue',kind='line', label='3 years ago')

ax2 = df_im2.twoyearsago.plot(color='red', kind='line', label='2 years ago')

ax3 = df_im2.lastyear.plot(color='green', kind='line', label='last year')



plt.show()
#create dataframe of latitude and longitude

df_loc = dfa3[['date','latitude','longitude','injured']]



df_loc = df_loc[pd.DatetimeIndex(df_loc.date).year == currentyear]



df_loc = df_loc[df_loc.injured > 0]



df_loc.head(1)
#Access token from Plotly

mapbox_access_token = 'pk.eyJ1Ijoia3Jwb3BraW4iLCJhIjoiY2pzcXN1eDBuMGZrNjQ5cnp1bzViZWJidiJ9.ReBalb28P1FCTWhmYBnCtA'



#Prepare data for Plotly

data = [

    go.Scattermapbox(

        lat=df_loc.latitude,

        lon=df_loc.longitude,

        mode='markers',

        text=df_loc.injured,

        marker=dict(

            size=7,

            color=df_loc.injured,

            colorscale='RdBu',

            reversescale=True,

            colorbar=dict(

                title='Injured'

            )

        ),

    )

]
#Prepare layout for Plotly

layout = go.Layout(

    autosize=True,

    hovermode='closest',

    title='NYPD Motor Vehicle Collisions in ' + str(currentyear),

    mapbox=dict(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=40.721319,

            lon=-73.987130

        ),

        pitch=0,

        zoom=11

    ),

)
#Create map using Plotly

fig = dict(data=data, layout=layout)

iplot(fig, filename='NYPD Collisions')