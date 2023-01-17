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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import chart_studio.plotly as py

import plotly.express as px

import plotly.graph_objects as go
from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
newyork = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
newyork.head()
newyork.info()
newyork.describe()
newyork.isnull().sum()
newyork.drop(['id','host_name','last_review','reviews_per_month'],axis=1,inplace=True)
newyork.head()
newyork.dropna(inplace=True)
newyork.info()
plt.figure(figsize=(10,5))

sns.set_style('darkgrid')

sns.countplot(x='neighbourhood_group',data=newyork,palette='plasma',order=newyork['neighbourhood_group'].value_counts().index)

plt.title('Airbnb Listings by Borough')
plt.figure(figsize=(10,10))

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', data=newyork)

sns.set_style('whitegrid')
mb_token = 'pk.eyJ1IjoiZXNpZXJyMSIsImEiOiJjazVlZzRyamowNG8xM3BwYnVxeGdncGI2In0.NUtfxve0tVtWfMjowlIHUg'
manhattan = newyork[newyork['neighbourhood_group'] == 'Manhattan']

brooklyn = newyork[newyork['neighbourhood_group'] == 'Brooklyn']

queens = newyork[newyork['neighbourhood_group'] == 'Queens']

bronx = newyork[newyork['neighbourhood_group'] == 'Bronx']

staten = newyork[newyork['neighbourhood_group'] == 'Staten Island']



data = [go.Scattermapbox(

            lat=manhattan['latitude'],

            lon=manhattan['longitude'],

            mode='markers',

            marker=dict(

                size= 5,

                color = 'blue',

                opacity = .8),

            name ='Manhattan'

          ),

        go.Scattermapbox(

            lat=brooklyn['latitude'],

            lon=brooklyn['longitude'],

            mode='markers',

            marker=dict(

                size= 5,

                color = 'red',

                opacity = .8),

            name ='Brooklyn'

          ),

       go.Scattermapbox(

            lat=queens['latitude'],

            lon=queens['longitude'],

            mode='markers',

            marker=dict(

                size= 5,

                color = 'yellow',

                opacity = .8),

            name ='Queens'

         ),

       go.Scattermapbox(

            lat=bronx['latitude'],

            lon=bronx['longitude'],

            mode='markers',

            marker=dict(

                size= 5,

                color = 'green',

                opacity = .8),

            name ='Bronx'

         ),

       go.Scattermapbox(

            lat=staten['latitude'],

            lon=staten['longitude'],

            mode='markers',

            marker=dict(

                size= 5,

                color = 'orange',

                opacity = .8),

            name ='Staten Island')]



layout = go.Layout(autosize=True,

                   title="Airbnb Listings by Borough",

                   mapbox=dict(accesstoken=mb_token,

                               bearing=0,

                               pitch=50,

                               zoom=9,

                               center=dict(

                                   lat=40.6782,

                                   lon=-73.9442)

                               ))





fig = dict(data=data, layout=layout)

iplot(fig)
plt.figure(figsize=(10,5))

sns.set_style('darkgrid')

sns.countplot(x='room_type',data=newyork,palette='coolwarm',order=newyork['room_type'].value_counts().index)

plt.title('Airbnb Listings by Type')
plt.figure(figsize=(10,10))

sns.scatterplot(x='longitude', y='latitude', hue='room_type', data=newyork)

sns.set_style('whitegrid')
private_room = newyork[newyork['room_type'] == 'Private room']

entire_home = newyork[newyork['room_type'] == 'Entire home/apt']

shared_room = newyork[newyork['room_type'] == 'Shared room']



data = [go.Scattermapbox(

            lat=private_room['latitude'],

            lon=private_room['longitude'],

            mode='markers',

            marker=dict(

                size= 5,

                color = 'blue',

                opacity = .8),

            name ='Private room'

          ),

        go.Scattermapbox(

            lat=entire_home['latitude'],

            lon=entire_home['longitude'],

            mode='markers',

            marker=dict(

                size= 5,

                color = 'red',

                opacity = .8),

            name ='Entire home/apt'

          ),

       go.Scattermapbox(

            lat=shared_room['latitude'],

            lon=shared_room['longitude'],

            mode='markers',

            marker=dict(

                size= 5,

                color = 'yellow',

                opacity = .8),

            name ='Shared room')]



layout = go.Layout(autosize=True,

                   title="Airbnb Listings by Type",

                   mapbox=dict(accesstoken=mb_token,

                               bearing=0,

                               pitch=50,

                               zoom=9,

                               center=dict(

                                   lat=40.6782,

                                   lon=-73.9442)

                               ))





fig = dict(data=data, layout=layout)

iplot(fig)
newyork['neighbourhood'].value_counts().head(10)
plt.figure(figsize=(10,5))

sns.set_style('darkgrid')

sns.countplot(y='neighbourhood',data=newyork,palette='coolwarm',order=newyork['neighbourhood'].value_counts().index[:10])

plt.title('NYC Neighborhoods With Most Airbnb Listings')
williamsburg = newyork[newyork['neighbourhood']=='Williamsburg'].nlargest(10, ['price'])

bedstuy = newyork[newyork['neighbourhood']=='Bedford-Stuyvesant'].nlargest(10, ['price'])

harlem = newyork[newyork['neighbourhood']=='Harlem'].nlargest(10, ['price'])

bushwick = newyork[newyork['neighbourhood']=='Bushwick'].nlargest(10, ['price'])

upper_west = newyork[newyork['neighbourhood']=='Upper West Side'].nlargest(10, ['price'])

hells_kitchen = newyork[newyork['neighbourhood']=="Hell's Kitchen"].nlargest(10, ['price'])

east_village = newyork[newyork['neighbourhood']=='East Village'].nlargest(10, ['price'])

upper_east = newyork[newyork['neighbourhood']=='Upper East Side'].nlargest(10, ['price'])

crown_heights = newyork[newyork['neighbourhood']=='Crown Heights'].nlargest(10, ['price'])

midtown = newyork[newyork['neighbourhood']=='Midtown'].nlargest(10, ['price'])
data = [go.Scattermapbox(

            lat=williamsburg['latitude'],

            lon=williamsburg['longitude'],

            mode='markers',

            marker=dict(

                size= 7,

                color = 'rgb(0,75,141)',

                opacity = .8),

            name = 'Williamsburg',

            text= williamsburg['name']

          ),

        go.Scattermapbox(

            lat=bedstuy['latitude'],

            lon=bedstuy['longitude'],

            mode='markers',

            marker=dict(

                size= 7,

                color = 'rgb(84,135,164)',

                opacity = .8),

            name ='Bedford-Stuyvesant',

            text= bedstuy['price']

          ),

       go.Scattermapbox(

            lat=harlem['latitude'],

            lon=harlem['longitude'],

            mode='markers',

            marker=dict(

                size= 7,

                color = 'rgb(90,114,71)',

                opacity = .8),

            name ='Harlem',

            text= harlem['price']

         ),

       go.Scattermapbox(

            lat=bushwick['latitude'],

            lon=bushwick['longitude'],

            mode='markers',

            marker=dict(

                size= 7,

                color = 'rgb(246,209,85)',

                opacity = .8),

            name ='Bushwick',

            text= bushwick['price']

         ),

       go.Scattermapbox(

            lat=upper_west['latitude'],

            lon=upper_west['longitude'],

            mode='markers',

            marker=dict(

                size= 7,

                color = 'rgb(242,85,44)',

                opacity = .8),

            name ='Upper West Side',

            text= upper_west['price']

         ),

       go.Scattermapbox(

            lat=hells_kitchen['latitude'],

            lon=hells_kitchen['longitude'],

            mode='markers',

            marker=dict(

                size= 7,

                color = 'rgb(255,219,92)',

                opacity = .8),

            name ="Hell's Kitchen",

            text= hells_kitchen['price']

         ),

       go.Scattermapbox(

            lat=east_village['latitude'],

            lon=east_village['longitude'],

            mode='markers',

            marker=dict(

                size= 7,

                color = 'rgb(254,183,148)',

                opacity = .8),

            name ='East Village',

            text= east_village['price']

         ),

       go.Scattermapbox(

            lat=upper_east['latitude'],

            lon=upper_east['longitude'],

            mode='markers',

            marker=dict(

                size= 7,

                color = 'rgb(76,195,248)',

                opacity = .8),

            name ='Upper East Side',

            text= upper_east['price']

         ),

       go.Scattermapbox(

            lat=crown_heights['latitude'],

            lon=crown_heights['longitude'],

            mode='markers',

            marker=dict(

                size= 7,

                color = 'rgb(123,119,200)',

                opacity = .8),

            name ='Crown Heights',

            text= crown_heights['price']

         ),

       go.Scattermapbox(

            lat=midtown['latitude'],

            lon=midtown['longitude'],

            mode='markers',

            marker=dict(

                size= 7,

                color = 'rgb(254,101,148)',

                opacity = .8),

            name = 'Midtown',

            text= midtown['price']

         )]



layout = go.Layout(autosize=True,

                   title="NYC Neighborhoods With Most Listings",

                   mapbox=dict(accesstoken=mb_token,

                               bearing=0,

                               pitch=0,

                               zoom=9,

                               center=dict(

                                   lat=40.6782,

                                   lon=-73.9442)

                               ))





fig = dict(data=data, layout=layout)

iplot(fig)