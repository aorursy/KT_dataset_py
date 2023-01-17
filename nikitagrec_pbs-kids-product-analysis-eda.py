import pandas as pd

import numpy as np

import json

import seaborn as sns

import datetime

import matplotlib.pylab as plt

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from scipy.ndimage.filters import gaussian_filter

import warnings

import random

import plotly.express as px

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")

my_pal = sns.color_palette(n_colors=10)
plt.figure(figsize=(13,13))

image = plt.imread("/kaggle/input/pbs-images/IMAGE 2019-10-25 160313.jpg");

plt.imshow(image);

plt.axis('off');
n = 11341042 #number of records in file

s = 1000000 #desired sample size

filename = '../input/data-science-bowl-2019/train.csv'

skip = sorted(random.sample(range(n),n-s))

train_sam = pd.read_csv(filename, skiprows=skip)

train_sam.columns = ['event_id','game_session','timestamp','event_data',

            'installation_id','event_count','event_code','game_time','title','type','world']

specs = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
train_sam.head(3)
tips = train_sam[['game_session','installation_id']].drop_duplicates().groupby(['installation_id']).count().reset_index()

fig = px.histogram(tips[tips['game_session'] < 100], x="game_session", title='Sessions by user')

fig.show()
tips = train_sam[['game_session','event_count']].groupby(['game_session']).max().reset_index()

fig = px.histogram(tips[tips['event_count'] < 100], x="event_count", title='Event count by session')

fig.show()



tips = train_sam[['game_session','event_count']].groupby(['game_session']).max().reset_index()

fig = px.histogram(tips[(tips['event_count'] > 1) & (tips['event_count'] < 200)], x="event_count", 

                   title='Event count by session', nbins=40)

fig.show()
train_sam['date'] = train_sam['timestamp'].apply(lambda x: x.split('T')[0])

train_sam['date'] = train_sam['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
ss = train_sam[['date','installation_id']].groupby(['date']).count().reset_index()

ss.columns = ['date', 'activity_count']

fig = px.line(ss, x='date', y='activity_count', title='Activity')

fig.show()
temp = train_sam[['date','installation_id']].groupby(['installation_id']).min().reset_index()

ss = temp.groupby(['date']).count().reset_index()

ss.columns = ['date', 'installs_count']

fig = px.line(ss, x='date', y='installs_count', title='Installs')

fig.show()
fig = px.bar(x=pd.value_counts(train_sam['world']).index, y=pd.value_counts(train_sam['world']).values)

fig.show()
tips = train_sam[['game_session','type','event_count','world']].groupby(['type','world','game_session']).max().reset_index()

fig = px.histogram(tips[(tips['event_count']>1) & (tips['event_count']<100)], x="event_count", 

             facet_row="world", facet_col="type", nbins=40,

             category_orders={"world": ['CRYSTALCAVES', 'MAGMAPEAK', 'NONE', 'TREETOPCITY'],

                             'type': ['Activity', 'Game', 'Clip', 'Assessment']})

fig.show()
event = pd.value_counts(train_sam.event_id)

event[event>17000].plot('barh', title='Popular events');
popular = specs.merge(event[event>17000], how='inner', right_index=True, left_on='event_id')

popular = popular[['info','event_id_y']].sort_values(by='event_id_y', ascending=False)

for i in range(5):

    print(i, popular['info'].iloc[i])

    print('________________________________________')
sec = train_sam[train_sam.event_id == '1325467d']

sec['xx'] = sec['event_data'].apply(lambda x: json.loads(x)['coordinates']['x'])

sec['yy'] = sec['event_data'].apply(lambda x: json.loads(x)['coordinates']['y'])
plt.figure(figsize=(10,10))

plt.scatter(sec['xx'], sec['yy']);
fig = plt.figure(figsize=(20,15))

for i, event in enumerate(['1325467d','cf82af56','cfbd47c8','76babcde','6c517a88','884228c8']):

    fig.add_subplot(2,3,i+1)

    kk = train_sam[train_sam.event_id == event]

    kk['xx'] = kk['event_data'].apply(lambda x: json.loads(x)['coordinates']['x'])

    kk['yy'] = kk['event_data'].apply(lambda x: json.loads(x)['coordinates']['y'])

    plt.hist2d(kk['xx'],kk['yy'], bins=60, cmap=plt.cm.jet)