### load required packages 

import pandas as pd 

import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')
### load the dataset 

df = pd.read_csv('../input/cameras.csv', encoding='utf-8')



## overview 

print('The shape:')

print(df.shape)



print('\nThe information:')

print(df.info())



# the number and percentage of NAs in each column

print('\nNAs:')

print(np.sum(df.isnull()))

print(np.sum(df.isnull())/len(df)*100) # in percentage



# the start and the end of the date

print('\nStart and End:')

print(df['DATE'][[0, df.shape[0]-1]])



# the number of different cameras

print('\nDifferent cameras:')

print(df['CAMERA ID'].value_counts().head())

print(len(df['CAMERA ID'].value_counts()))
### DATE



## Convert the DATE column into datetime object 

from datetime import datetime



df['DATE'] = df['DATE'].map(lambda x: datetime.strptime(x, '%m/%d/%Y'))

print(df['DATE'].head()) # for checking 



## Get the months and weekdays

def week_day(val):

    week_days_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    return week_days_list[val]



df['Month'] = df['DATE'].map(lambda x: x.month)

df['Weekday'] = df['DATE'].map(lambda x: week_day(x.weekday()))



## If every day is in the data?

print((df['DATE'][len(df['DATE'])-1] - df['DATE'][0]).days+1)

print(len(df['DATE'].value_counts()))

print('Yes! Each day is in this dataset.')
## if the NAs are conventrated on certain dates?

ind_location_nas = np.argwhere(df['LOCATION'].isnull()).ravel()



# for dates

print('Number of days having NAs:')

print(len(df['DATE'][ind_location_nas].value_counts()))

fig, axes = plt.subplots(figsize=[12, 5])

df['DATE'][ind_location_nas].value_counts().sort_index().plot(color='#2E8B57')

axes.set_ylim([0, 10])

axes.set_title('NAs on Dates', fontsize=20)

axes.set_ylabel('Frequency')



# for months

fig, axes = plt.subplots(figsize=[12, 5])

df['Month'][ind_location_nas].value_counts().sort_index().plot(color='#2E8B57')

axes.set_title('NAs on Months', fontsize=20)

axes.set_ylabel('Frequency')



# for weekdays

weekday_count = dict(df['Weekday'][ind_location_nas].value_counts())

weekday_count_list = []

week_days_list = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

for day in week_days_list:

    weekday_count_list.append(weekday_count[day])



fig, axes = plt.subplots(figsize=[12, 5])

plt.plot(weekday_count_list, color='#2E8B57')

axes.set_xticks(np.arange(len(week_days_list)))

axes.set_xticklabels(week_days_list)

axes.set_title('NAs on Weekdays', fontsize=20)

axes.set_ylabel('Frequency')



## Check the ratio



# for dates

date_group = df.groupby(['DATE'])

fig, axes = plt.subplots(figsize=[12, 5])

(df['DATE'][ind_location_nas].value_counts().sort_index()/date_group['VIOLATIONS'].sum()*100).sort_index().plot(color='#2E8B57')

axes.set_title('NAs on Dates (in %)', fontsize=20)

axes.set_ylabel('Ratio (in %)')

print('The highest:')

print(np.argmax((df['DATE'][ind_location_nas].value_counts().sort_index()/date_group['VIOLATIONS'].sum()*100)))



# for months

month_group = df.groupby(['Month'])

fig, axes = plt.subplots(figsize=[12, 5])

(df['Month'][ind_location_nas].value_counts().sort_index()/month_group['VIOLATIONS'].sum()*100).sort_index().plot(color='#2E8B57')

axes.set_title('NAs on Months (in %)', fontsize=20)

axes.set_ylabel('Ratio (in %)')



# for weekdays

week_group = df.groupby(['Weekday'])

weekday_violations = dict(week_group['VIOLATIONS'].sum())



weekday_violatioin_ratio_list = []

for day in week_days_list:

    weekday_violatioin_ratio_list.append(weekday_count[day]/weekday_violations[day]*100)



fig, axes = plt.subplots(figsize=[12, 5])

plt.plot(weekday_violatioin_ratio_list, color='#2E8B57')

axes.set_xticks(np.arange(len(week_days_list)))

axes.set_xticklabels(week_days_list)

axes.set_title('NAs on Weekdays (in %)', fontsize=20)

axes.set_ylabel('Ratio (in %)')    
### CAMERA



## How many different cameras are there?

print('Number of different cameras:')

print(len(df['CAMERA ID'].value_counts()))

print()

print('Number of differenct addresses:')

print(len(df['ADDRESS'].value_counts()))



print()

print('Some cameras have been moved?')
## check the un-matched cameras and addresses

camera_group = df.groupby(['CAMERA ID'])

print(camera_group['ADDRESS'].unique().head())



print()

print('The camera that has multiple addresses:')

for camera in list(dict(camera_group['ADDRESS'].unique()).keys()):

    if len(dict(camera_group['ADDRESS'].unique())[camera]) > 1:

        print(camera)

print(dict(camera_group['ADDRESS'].unique())['CHI126'])



print()

print(df.ix[df['CAMERA ID'] == 'CHI126', ['DATE', 'ADDRESS', 'LATITUDE', 'LONGITUDE', 'LOCATION']].head())

print(df.ix[df['CAMERA ID'] == 'CHI126', ['DATE', 'ADDRESS', 'LATITUDE', 'LONGITUDE', 'LOCATION']].tail())
ind_location_nas = np.argwhere(df['LOCATION'].isnull()).ravel()



print(df['CAMERA ID'][ind_location_nas].value_counts())

print('The NAs concentrate on the above five cameras')



df['CAMERA ID'][ind_location_nas].value_counts().plot(kind='bar', color='#2E8B57', edgecolor='none', rot=0)

plt.title('NAs on cameras')

plt.ylabel('Frequency')

plt.xlabel('Camera')
## for dates

date_group = df.groupby(['DATE'])

fig, axes = plt.subplots(figsize=[12, 5])

date_group['VIOLATIONS'].sum().sort_index().plot(color='#2E8B57')

axes.set_title('Violations on Date', fontsize=25)

axes.tick_params(labelsize='large')

axes.set_ylabel('Frequency', fontsize=20)

axes.set_xlabel('Date', fontsize=20)



# There seems to be an outlier on 2015/02/01

print('Outlier:')

print(np.argmin(date_group['VIOLATIONS'].sum()))

print(date_group['VIOLATIONS'].sum().min())



## for months

month_group = df.groupby(['Month'])

fig, axes = plt.subplots(figsize=[12, 5])

month_group['VIOLATIONS'].sum().sort_index().plot(color='#2E8B57')

axes.set_title('Violations on Month', fontsize=25)

axes.tick_params(labelsize='large')

axes.set_ylabel('Frequency', fontsize=20)

axes.set_xlabel('Month', fontsize=20)



## for weekdays

week_group = df.groupby(['Weekday'])

weekday_violations = dict(week_group['VIOLATIONS'].sum())



weekday_violatioin_list = []

for day in week_days_list:

    weekday_violatioin_list.append(weekday_violations[day])



fig, axes = plt.subplots(figsize=[12, 5])

plt.plot(weekday_violatioin_list, color='#2E8B57')

axes.set_xticks(np.arange(len(week_days_list)))

axes.set_xticklabels(week_days_list)

axes.set_title('Violationis on Weekdays', fontsize=25)

axes.tick_params(labelsize='large')

axes.set_ylabel('Frequency', fontsize=20)

axes.set_xlabel('Weekday', fontsize=20)
## Does each camera have the same records?

print(df['CAMERA ID'].value_counts().head())

print(df['CAMERA ID'].value_counts().tail())

print('The differences of records of each camera are large')



print()

print('The number of records for the camera with NAs in location-related data:')

print(df['CAMERA ID'].value_counts()[list(df['CAMERA ID'][ind_location_nas].value_counts().index)])
## Drop the NAs

df_nna = df.dropna(how='any')

df_nna.index = range(len(df_nna))

print(df_nna.shape)

print(len(df_nna['CAMERA ID'].value_counts())) # 5 cameras are excluded
## Use the new location for CHI126 to replace the original one

new_location = list(df_nna.ix[len(df_nna)-1, ['LATITUDE', 'LONGITUDE', 'LOCATION']].values)

print('New location for CHI126:')

print(new_location)



for ind, col in enumerate(['LATITUDE', 'LONGITUDE', 'LOCATION']):

    df_nna.ix[df_nna['CAMERA ID'] == 'CHI126', col] = df_nna.ix[df_nna['CAMERA ID'] == 'CHI126', col].apply(lambda x: new_location[ind])



print()

print('Checking:')

print(df_nna.ix[df_nna['CAMERA ID'] == 'CHI126', ['LATITUDE', 'LONGITUDE', 'LOCATION']].head())



print()

print('Number of different locations:')

print(len(df_nna['LOCATION'].value_counts()))



print()

print('There are same locations?') 

print(df_nna['LOCATION'].value_counts().head()) # (41.742993, -87.6611378)

print(df_nna.ix[df_nna['LOCATION'] == '(41.742993, -87.6611378)', :]['CAMERA ID'].value_counts()) # CHI 126 and CHI 170

#print(df_nna.ix[df_nna['CAMERA ID'].isin(['CHI126', 'CHI170']), :])

print(df_nna.ix[[264, 89408], :]) # They have different addresses; however the locations are very near
## Get the data for sum of violations and location



# use the 2016 data

camera_loc_group = df_nna.ix[df['DATE'] > datetime(2015, 12, 31), :].groupby(['CAMERA ID', 'LATITUDE', 'LONGITUDE'])

camera_loc_vio = dict(camera_loc_group['VIOLATIONS'].sum())



from collections import defaultdict

temp_tab = defaultdict(list)

for key in list(camera_loc_vio.keys()):

    temp_tab[key[0]].append(camera_loc_vio[key])

    temp_tab[key[0]].extend(key[1:])

    

vio_loc_df = pd.DataFrame(temp_tab).transpose()

vio_loc_df.columns = ['VIOLATIONS', 'lat', 'lon']

print(vio_loc_df.head())
## Visualization



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly.graph_objs import *

import plotly.graph_objs as go



mapbox_access_token = 'pk.eyJ1IjoieG5pcGVyIiwiYSI6ImNqMDR6cXR0aDBoNm4ycWxzcTF2Z3ZxbGsifQ.dAlvq0ZttViD4l3HRbqeYw'



data = Data([

    Scattermapbox(

        lat=list(vio_loc_df['lat'].values),

        lon=list(vio_loc_df['lon'].values),

        mode='markers',

        marker=Marker(

            size=8,

            color='#000000',

            opacity=1

        ),

        text=list(vio_loc_df.index),

    ),

        Scattermapbox(

        lat=list(vio_loc_df['lat'].values),

        lon=list(vio_loc_df['lon'].values),

        mode='markers',

        marker=Marker(

            size=5,

            color='#FFFF33',

            opacity=0.7

        ),

        text=list(vio_loc_df.index),hoverinfo='skip'

    )])



layout = Layout(

    showlegend=False,

    autosize=False,

    width=600,

    height=1000,

    title='The Distribution of Cameras',

    hovermode='closest',

    margin=go.Margin(

        l=50,

        r=10,

        b=50,

        t=100,

        pad=2

    ),

    mapbox=dict(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=np.mean(vio_loc_df['lat']),

            lon=np.mean(vio_loc_df['lon'])

        ),

        pitch=0,

        zoom=10

    ),

)



fig = dict(data=data, layout=layout)

iplot(fig, filename='Camera_distribution', validate=False)
## Visualization



# scale the violations to 0 to 1

vio_loc_df['VIOLATIONS_scl'] = vio_loc_df['VIOLATIONS'].apply(lambda x: (x-vio_loc_df['VIOLATIONS'].min())/(vio_loc_df['VIOLATIONS'].max()-vio_loc_df['VIOLATIONS'].min()))



mapbox_access_token = 'pk.eyJ1IjoieG5pcGVyIiwiYSI6ImNqMDR6cXR0aDBoNm4ycWxzcTF2Z3ZxbGsifQ.dAlvq0ZttViD4l3HRbqeYw'



scl = [[0, 'rgb(2, 223, 132)'], [1, 'rgb(0, 0, 0)']]



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly.graph_objs import *

import plotly.graph_objs as go



mapbox_access_token = 'pk.eyJ1IjoieG5pcGVyIiwiYSI6ImNqMDR6cXR0aDBoNm4ycWxzcTF2Z3ZxbGsifQ.dAlvq0ZttViD4l3HRbqeYw'



data = Data([    



    Scattermapbox(

        lat=list(vio_loc_df['lat'].values),

        lon=list(vio_loc_df['lon'].values),

        mode='markers',

        marker=Marker(

            size=15,

            color=list(vio_loc_df['VIOLATIONS'].values),

            opacity=0.9,

                        colorscale=scl,

        cmin=0,

        cmax=vio_loc_df['VIOLATIONS'].values.max(),

        colorbar=dict()

        ),

        text=list(vio_loc_df.index),

    ),])



layout = Layout(

    showlegend=False,

    autosize=False,

    width=800,

    height=1000,

    title='The Distribution of Number of Violatioins Captured (in 2016)',

    hovermode='closest',

    margin=go.Margin(

        l=50,

        r=10,

        b=50,

        t=100,

        pad=2

    ),

    mapbox=dict(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=np.mean(vio_loc_df['lat']),

            lon=np.mean(vio_loc_df['lon'])

        ),

        pitch=0,

        zoom=10

    ),

)



fig = dict(data=data, layout=layout)

iplot(fig, filename='Violation_distribution', validate=False)