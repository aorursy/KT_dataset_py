# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/formula-1-german-grandprix-2019-race-day-data/race_classification_germanGp-2019.csv')
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
data.head()
teams=data.copy()

del teams['TEAM']

s = data['TEAM'].str.split(',').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1)

s.name = 'TEAM'

teams = teams.join(s)
lis=[]

for i in teams['TEAM']:

    lis.append(i)



for k in range(0,len(lis)):

    lis[k]=str(lis[k]).strip()

    

from collections import Counter

genre_count = Counter(lis)



from wordcloud import WordCloud

wc = WordCloud(background_color='white')

wc.generate_from_frequencies(genre_count)

plt.figure(figsize=(20,10))

plt.imshow(wc,interpolation='bilinear')

plt.axis('off')

plt.show()
drivers=data.copy()

del drivers['NAME']

s = data['NAME'].str.split(',').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1)

s.name = 'NAME'

drivers = drivers.join(s)
lis=[]

for i in drivers['NAME']:

    lis.append(i)



for k in range(0,len(lis)):

    lis[k]=str(lis[k]).strip()

    

from collections import Counter

genre_count = Counter(lis)



from wordcloud import WordCloud

wc = WordCloud(background_color='black')

wc.generate_from_frequencies(genre_count)

plt.figure(figsize=(20,10))

plt.imshow(wc,interpolation='bilinear')

plt.axis('off')

plt.show()
data.head()
fig = px.bar(data, x='NAME', y='STARTING_POSITION', color='NAME', height=600, title='Grid positions')

fig.show()
fig = px.bar(data, x='NAME', y='RACE_PLACE', color='NAME', height=600, title='Places finished')

fig.show()
fig = px.bar(data, x='NAME', y='PLACES_GAINED', color='NAME', height=600, title='Places gained')

fig.show()
points_score= data.sort_values(by = 'POINTS', ascending = False).head(20)

fig = px.sunburst(points_score, path= ['POINTS','NAME'])

fig.show()

fig = px.bar(data, x='NAME', y='FASTEST_LAP', color='NAME', height=600, title='fastest laps')

fig.show()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

fig = px.pie(data,names='NAME', values='SECTOR1')

fig.update_traces(rotation=45, pull=[0.1,0.03,0.03,0.03,0.03],textinfo="label", title='Fastest sector 1 times')

fig.show()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

fig = px.pie(data,names='NAME', values='SECTOR2 ')

fig.update_traces(rotation=45, pull=[0.1,0.03,0.03,0.03,0.03],textinfo="label", title='Fastest sector 2 times')

fig.show()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

fig = px.pie(data,names='NAME', values='SECTOR3')

fig.update_traces(rotation=45, pull=[0.1,0.03,0.03,0.03,0.03],textinfo="label", title='Fastest sector 3 times')

fig.show()
points_score= data.sort_values(by = 'PIT_STOPS', ascending = False).head(20)

fig = px.treemap(points_score, path= ['PIT_STOPS','NAME'])

fig.show()
