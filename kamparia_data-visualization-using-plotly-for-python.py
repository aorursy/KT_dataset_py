# imports pandas, numpy and matplotlib modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# import plotly modules
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')
## load csv data
file_path = '../input/autos.csv'
df = pd.read_csv(file_path, encoding='iso8859_2')
## remove all NAs
df = df.dropna(subset=['name', 'seller', 'offerType', 'price', 'abtest', 'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType', 'brand', 'notRepairedDamage', 'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen'], how='any')

## subset multiple columns of our dataframe
df = df[['name', 'seller', 'offerType', 'price', 'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 'model', 'kilometer', 'fuelType', 'brand', 'notRepairedDamage', 'nrOfPictures', 'postalCode']] 

## remove rows where price is less than 500
df = df[df['price'] >= 100]

## translate data from german to english
df['seller'] = df['seller'].replace(to_replace=['privat', 'gewerblich'], value=['private', 'commercial'], inplace=False, limit=None)
df['offerType'] = df['offerType'].replace(to_replace=['Angebot'], value=['offer'], inplace=False, limit=None)
df['gearbox'] = df['gearbox'].replace(to_replace=['automatik', 'manuell'], value=['automatic', 'manual'], inplace=False, limit=None)
df['fuelType'] = df['fuelType'].replace(to_replace=['andere', 'benzin', 'elektro'], value=['others', 'petrol', 'electric'], inplace=False, limit=None)
df['vehicleType'] = df['vehicleType'].replace(to_replace=['andere', 'kombi', 'kleinwagen'], value=['others', 'station wagon', 'small car'], inplace=False, limit=None)
df['notRepairedDamage'] = df['notRepairedDamage'].replace(to_replace=['ja', 'nein'], value=['yes', 'no'], inplace=False, limit=None)
# 1. Visualization of dataset based year of registration.
df1 = df['yearOfRegistration'].value_counts()
trace = go.Bar(
    x=df1.index,
    y=df1.values,
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),    
    opacity=0.6
)

data = [trace]

layout = go.Layout(
    title='',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='used_cars_year')
# 2. Visualization of dataset based on gearbox of the vehicle.
df2 = df['gearbox'].value_counts()
colors = ['#FEBFB3', '#E1396C']

trace = go.Pie(
    labels=df2.index, 
    values=df2.values,
    hoverinfo='label+percent', 
    textinfo='value', 
    textfont=dict(size=20),
    marker=dict(colors=colors, 
        line=dict(color='#000000', width=2)
    )
)

data = [trace]

layout = go.Layout(
    title='',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='used_cars_gearbox')
# 3. Visualization of dataset based on vehicle type.
df3 = df['vehicleType'].value_counts()
trace = go.Bar(
    x=df3.index,
    y=df3.values,
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),    
    opacity=0.6
)

data = [trace]

layout = go.Layout(
    title='',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='used_cars_vehicleType')
# 4. Visualization of dataset based on average price of cars per brand.
df4 = df.groupby(['brand'])['price'].mean().reset_index()
trace = go.Bar(
    x=df4['brand'],
    y=df4['price'],
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),    
    opacity=0.6
)

data = [trace]

layout = go.Layout(
    title='',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='used_cars_avg_brand_price')
# 5. Visualization of dataset based on fuel types of cars.
df5 = df['fuelType'].value_counts()
trace = go.Bar(
    x=df5.index,
    y=df5.values,
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),    
    opacity=0.6
)

data = [trace]

layout = go.Layout(
    title='',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='used_cars_fuelType')
# 6. Visualization of dataset based on the gearbox of cars.
df6_manual = df[df['gearbox'] == 'manual']['brand'].value_counts().reset_index().rename(columns=dict(zip(['brand'], ['manual'])))
df6_automatic = df[df['gearbox'] == 'automatic']['brand'].value_counts().reset_index().rename(columns=dict(zip(['brand'], ['automatic'])))
df6 = pd.merge(df6_manual, df6_automatic, left_on=['index'], right_on=['index'], how='inner').sort_values('index', ascending=True)

trace1 = go.Bar(
    x=df6['index'],
    y=df6['manual'],
    name='Manual'
)
trace2 = go.Bar(
    x=df6['index'],
    y=df6['automatic'],
    name='Automatic'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='stacked-gearbox')
