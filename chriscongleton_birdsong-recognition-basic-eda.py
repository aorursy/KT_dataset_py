import numpy as np

import pandas as pd

import plotly.express as px
train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')

train.head()
ratings = train['rating'].value_counts().to_frame()

ratings.rename(columns={'rating':'Instances'}, inplace=True)

ratings['Rating'] = ratings.index

fig = px.bar(ratings, x='Rating', y='Instances',

            labels={'Instances':'Instances in train'} )



fig.show()
species = train['species'].value_counts().to_frame()

fig = px.bar(species, x=species.index, y='species',

            labels={'species:Species of birds'})

fig.show()
channels = train['channels'].value_counts().to_frame()

channels.rename(columns={'channels':'Instances'}, inplace=True)

channels['Channel'] = channels.index

channels.reset_index(drop=True, inplace=True)

channels.head()
train['month'] = train.date.str[5:7].astype(int)

months = train.month.value_counts().to_frame()

months.rename(columns={'month':'Instances'}, inplace=True)

months['Month'] = months.index

months.reset_index(drop=True, inplace=True)

months.head()
fig = px.bar(months, x='Month', y='Instances')



fig.update_layout(

    title="Distribution of bird calls recorded over the months"

)

fig.show()
look_up = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',

            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec', 0 : '13th'}

months.Month = months.Month.apply(lambda x: look_up[x])

fig = px.bar(months, x='Month', y='Instances')

fig.update_layout(

    title="Distribution of bird calls recorded over the months ordered by instances"

)

fig.show()
pitches = train['pitch'].value_counts().to_frame()

pitches.rename(columns={'pitch':'Instances'}, inplace=True)

pitches['pitch'] = pitches.index

pitches.reset_index(drop=True, inplace=True)

pitches.head()
fig = px.bar(pitches, x='pitch', y='Instances')

fig.update_layout(

    title="Distribution of bird call pitch",

    xaxis_title="Pitch"

)

fig.show()
fig = px.box(train, y='duration')

fig.update_layout(

    title="Duration of recordings",

    yaxis_title="Duration (seconds)"

)

fig.show()
times = train.time.value_counts().to_frame()

times.rename(columns={'time':'Instances'}, inplace=True)

times['time'] = times.index

times.reset_index(drop=True, inplace=True)
fig = px.bar(times, x='time', y='Instances')

fig.update_layout(

    title="Distribution of bird calls per time stamp",

    xaxis_title="Pitch"

)

fig.show()