import numpy as np

import pandas as pd



import plotly.express as px

import cufflinks as cf

from plotly.offline import download_plotlyjs , init_notebook_mode

init_notebook_mode(connected = True)

cf.go_offline()



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')

df.head()
df.info()
df.describe()
df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].apply(lambda x : x.year)

df['month'] = df['date'].apply(lambda x : x.month)

df['weekday'] = df['date'].apply(lambda x : x.weekday())

df.head()
ds = df['user_location'].value_counts().reset_index()

ds.columns = ['Location' , 'Count']

ds.sort_values(['Count'] , ascending = False)



fig = px.bar(

    ds[:20],

    x = 'Location',

    y = 'Count',

    color = 'Location',

    title = 'Users Location Distribution'

)

fig.update_layout(showlegend=False)

fig.show()
ds = df.groupby(['user_location' , 'weekday'])['user_name'].count().reset_index().sort_values('user_name' , ascending = False)

ds.columns = ['Location' ,  'Weekday' , 'Count']

ds = ds[ds['Weekday'] == 5]



fig = px.bar(

    ds[:20],

    x = 'Location',

    y = 'Count',

    color = 'Location',

    title = 'Users Location Distribution on Saturday Tweets',

)

fig.update_layout(showlegend=False)

fig.show()
ds = df.groupby(['user_location' , 'weekday'])['user_name'].count().reset_index().sort_values('user_name' , ascending = False)

ds.columns = ['Location' ,  'Weekday' , 'Count']

ds = ds[ds['Weekday'] == 6]



fig = px.bar(

    ds[:20],

    x = 'Location',

    y = 'Count',

    color = 'Location',

    title = 'Users Location Distribution on Sunday Tweets',

)

fig.update_layout(showlegend=False)

fig.show()
ds = df[df['user_verified'] == True]

ds = ds['user_location'].value_counts().reset_index()

ds.columns = ['Location' , 'Count']

ds.sort_values('Count' , ascending = False)



fig = px.bar(

    ds[:20],

    x = 'Location',

    y = 'Count',

    color = 'Location',

    title = 'Verified Users Location Distribution'

)

fig.update_layout(showlegend=False)

fig.show()
df['gadget'] = df['source'].str.split(' ').str[-1]





ds = df[df['gadget'] == 'Android']

ds = ds.groupby(['user_location'])['gadget'].count().reset_index()

ds.columns = ['Location' , 'Count']

ds = ds.sort_values('Count' , ascending = False)



fig = px.bar(

    ds[:20],

    x = 'Location',

    y = 'Count',

    color = 'Location',

    title = 'Most Tweets Sent by an Android Gadget per Country'

)

fig.update_layout(showlegend=False)

fig.show()
ds = df[df['gadget'] == 'iPhone']

ds = ds.groupby(['user_location'])['gadget'].count().reset_index()

ds.columns = ['Location' , 'Count']

ds = ds.sort_values('Count' , ascending = False)



fig = px.bar(

    ds[:20],

    x = 'Location',

    y = 'Count',

    color = 'Location',

    title = 'Most Tweets Sent by an iPhone Gadget per Country'

)

fig.update_layout(showlegend=False)

fig.show()
ds = df.groupby(['user_location'])['user_followers'].count().reset_index().sort_values('user_followers' , ascending = False)

ds.columns = ['Location' , 'Count']



fig = px.bar(

    ds[:20],

    x = 'Location',

    y = 'Count',

    color = 'Location',

    title = 'Users Who Followed by Most People per Location'

)

fig.update_layout(showlegend=False)



fig.show()