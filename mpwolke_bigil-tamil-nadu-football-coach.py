#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT0U36S_QeLqWr1X7y_EmPmxVVmarE_nNEmH67RHM3WUi2QCaVE',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import iplot

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/bigiltamil-nadu-football-coach/events.csv')



df.head()
df.dtypes
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.isnull().sum()
df.dropna(how = 'all',inplace = True)

df.drop(['event_type2','player_in', 'player_out', 'shot_place'],axis=1,inplace = True)

df.shape
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
dfcorr=df.corr()

dfcorr
sns.heatmap(df.corr())

plt.show()
sns.barplot(x=df['opponent'].value_counts().index,y=df['opponent'].value_counts())
trace1 = go.Box(

    y=df["side"],

    name = 'side',

    marker = dict(color = 'rgb(0,145,119)')

)



trace2 = go.Box(

    y=df["time"],

    name = 'time',

    marker = dict(color = 'rgb(5, 79, 174)')

)



data = [trace1, trace2]

layout = dict(autosize=False, width=700,height=500, title='Bigil', paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))



fig = dict(data=data, layout=layout)

iplot(fig)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.event_team)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQVuSJbtC9e4FF0JnNDAudS64aD5V0RyBhBmwjFu1NxdOYqKiGM',width=400,height=400)