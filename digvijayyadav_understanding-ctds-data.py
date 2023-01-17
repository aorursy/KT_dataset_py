!pip install dexplot
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import json

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder



#plotly

!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



import plotly.express as px

import dexplot as dxp

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
anchor_df = pd.read_csv('../input/chai-time-data-science/Anchor Thumbnail Types.csv')

youtube_df = pd.read_csv('../input/chai-time-data-science/YouTube Thumbnail Types.csv')

description_df = pd.read_csv('../input/chai-time-data-science/Description.csv')

episodes_df = pd.read_csv('../input/chai-time-data-science/Episodes.csv', parse_dates=['recording_date','release_date'])

performance_df = pd.read_csv('../input/chai-time-data-science/Results.csv')
anchor_df.head(5)
youtube_df.head(5)
description_df.head(5)
episodes_df.head(5)
performance_df.head(5)
episodes_df.info()
labels = episodes_df.heroes_gender.value_counts()[:15].index

values = episodes_df.heroes_gender.value_counts()[:15].values

colors = ['ligthblue', 'lightgreen']



plt.figure(figsize=(10,10))

fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values, marker=dict(colors=colors))])

fig.show()
labels = episodes_df.heroes_nationality.value_counts()[:15].index

values = episodes_df.heroes_nationality.value_counts()[:15].values

#colors = ['ligthblue', 'lightgreen']



plt.figure(figsize=(10,10))

fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values )])

fig.show()
dxp.count('category',episodes_df, figsize=(4,3),orientation='h')
episodes_df['episode_duration'].count()
dxp.count('apple_listeners',episodes_df, figsize=(10,5), split='heroes_gender', orientation='h')
dxp.count('category',episodes_df, figsize=(10,5), split='heroes_gender', orientation='h')
dxp.count('flavour_of_tea',episodes_df, figsize=(4,3), split = 'heroes_gender', orientation='h')
dxp.count('flavour_of_tea',episodes_df, figsize=(4,3), orientation='h')
labels = episodes_df.heroes.value_counts()[:15].index

values = episodes_df.heroes.value_counts()[:15].values



plt.figure(figsize=(10,10))

fig = go.Figure(data=[go.Pie(labels=labels, textinfo='label+percent', values=values )])

fig.show()
dxp.count('heroes_nationality',episodes_df, split='category',normalize=True,figsize=(10,6),size=0.9,stacked=True)
dxp.count('heroes_nationality',episodes_df, split='heroes_gender',normalize=True,figsize=(10,6),size=0.9,stacked=True)