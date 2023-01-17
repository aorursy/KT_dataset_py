import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
filename='/kaggle/input/top50spotify2019/top50.csv'

df=pd.read_csv(filename,encoding='ISO-8859-1')

df.head()


print(df.shape)
df.rename(columns={'Track.Name':'track_name','Artist.Name':'artist_name','Beats.Per.Minute':'beats_per_minute','Loudness..dB..':'Loudness_db','Valence.':'valence','Length.':'length', 'Acousticness..':'acousticness','Speechiness.':'speechiness'},inplace=True)

df.head()
import plotly.graph_objs as go



trace1 = go.Scatter(

        x = df.track_name,

        y = df.Danceability,

        mode = "lines+markers",

        name = "Danceability",

        marker = dict(color= 'rgba(16, 112, 2, 0.8)'),

        text = df.Danceability

)



trace2 = go.Scatter(

        x = df.track_name,

        y = df.beats_per_minute,

        mode = "lines",

        name = "Beats Per Minute",

        marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

        text = df.beats_per_minute

)

data = [trace1,trace2]

layout = dict(title = 'Top 50 Spotify Songs',

              xaxis=dict(title = 'Danceability',ticklen=5, zeroline = False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)