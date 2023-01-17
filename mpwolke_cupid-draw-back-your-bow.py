# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/okcupid-profiles/okcupid_profiles.csv', encoding='ISO-8859-2')



df.head()
!pip install vega_datasets
import altair as alt

from vega_datasets import data
alt.Chart(df.sample(1000)).mark_rect().encode(

    x='status',

    y='orientation',

    color='location',

    tooltip=['status','orientation','location']

).properties(width=550,height=380,title="Cupid Draw back your Bow")
Status=df[df['status']=='single']
alt.Chart(df.sample(1000)).mark_rect().encode(

    x='education',

    y='offspring',

    color=alt.Color('speaks',scale=alt.Scale(type='log',scheme='purples')),

    tooltip=['education','offspring', 'speaks']

).properties(width=600,height=400,title="Cupid Draw back your Bow")
Sign=df[(df['sign']=='taurus')&(df['status']!="single")]
alt.Chart(df.sample(1000)).mark_rect().encode(

    alt.X('age:Q', bin=alt.Bin(maxbins=3)),

    alt.Y('income:Q', bin=alt.Bin(maxbins=5)),

    alt.Color('count(height):Q', scale=alt.Scale(scheme='lightorange')),

).properties(width=600,height=400,title="Cupid Draw back your Bow")
Last_online=df[df['last_online']>'2012-03-19-17-41']
base = alt.Chart((df.sample(1000))).transform_aggregate(

    num_shows='count()',

    groupby=['location', 'last_online']

).encode(

    alt.X('location:O', scale=alt.Scale(paddingInner=0),title="Location"),

    alt.Y('last_online:O', scale=alt.Scale(paddingInner=0),title="Last Online"),

).properties(width=580,height=400,title="Cupid Draw back your Bow")



# Configure heatmap

heatmap = base.mark_rect().encode(

    color=alt.Color('num_shows:Q',

        scale=alt.Scale(scheme='darkgold'),

        legend=alt.Legend(direction='horizontal')

    )

)



# Configure text

text = base.mark_text(baseline='middle').encode(

    text='num_shows:Q',

    color=alt.condition(

        alt.datum.num_shows > 500,

        alt.value('black'),

        alt.value('white')

    )

)

# Draw the chart

heatmap + text
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'drugs', data = df, palette="rainbow",edgecolor="black")

plt.subplot(132)

plt.xticks(rotation=45)

sns.countplot(x= 'religion', data = df, palette="flag",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= 'drinks', data = df, palette="Greens_r",edgecolor="black")

plt.xticks(rotation=45)

plt.show()
cnt_srs = df['speaks'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Languages (Really?) Spoken Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="speaks")
cnt_srs = df['sign'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Reds',

        reversescale = True

    ),

)



layout = dict(

    title='Sign/Fun to think about it Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="sign")