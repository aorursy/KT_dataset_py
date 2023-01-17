import numpy as np

import pandas as pd

import os

import plotly.graph_objects as go
df = pd.read_excel('/kaggle/input/locale/localedata.xlsx')

print(df.columns)

data = df[['id','from_area_id', 'to_area_id', 'from_date','to_date','booking_created',

       'from_lat', 'from_long', 'to_lat', 'to_long',]]

data
# drop all rows with null lat long

data = data.dropna(subset=['from_lat', 'from_long'])



# Group by lat - lng. Done to get coordinates to plot   

aggregate_locs = data[['id', 'from_lat', 'from_long']].groupby(['from_lat', 'from_long']).count()



lat = []

lng = []

for from_lat, from_lng in aggregate_locs.index.to_list():

    lat.append(from_lat)

    lng.append(from_lng)
fig = go.Figure(go.Densitymapbox(lat=lat, lon=lng, z=aggregate_locs['id'], radius=15))

fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=180)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
agg = data.groupby([data['from_area_id'], data['from_date'].dt.hour])[['id']].count()

agg.rename(columns={'id':'trips'},inplace=True)

print(agg.describe())

agg.head()
agg.groupby(axis='index', level='from_date').sum().plot()
agg.groupby(axis='index', level='from_area_id').sum().reset_index().plot.scatter(x='from_area_id', y='trips')
start = agg.quantile(0.95)['trips']

high =  agg.quantile(1)['trips']



print('Starting surge from: ', start)

print('Highest trip count when drilled down on <area,hour> in data: ',high)
surges = np.arange(1.1, 2.6, 0.1)

print('Number of surge multipliers : ', len(surges))

print('Multipliers: ', surges)



interval = np.arange(1,len(surges)+1)

interval = start + interval * (high-start)/len(surges)

print('Intervals: ', interval)



def mapping(count):

    if count < start: return 1

    else:

        for i in range(len(interval)):

            if count <= interval[i]: return surges[i]

    return surges[-1]
agg['surge-metric'] = agg['trips'].apply(mapping)

agg.sample(20)