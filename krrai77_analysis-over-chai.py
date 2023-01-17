# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.express as px

import plotly.graph_objects as go

import matplotlib.pyplot as plt

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
episode=pd.read_csv("/kaggle/input/chai-time-data-science/Episodes.csv")

episode.head()
desc=pd.read_csv("/kaggle/input/chai-time-data-science/Description.csv")

desc.head()
episode.dtypes
episode['release_date']=pd.to_datetime(episode['release_date'])

episode['recording_date']=pd.to_datetime(episode['recording_date'])

episode['release_date']=episode['release_date'].dt.date

episode['recording_date']=episode['recording_date'].dt.date
episode.isna().sum()
episode.iloc[2]
fig = px.pie(episode, names='heroes_gender', title='Speaker Gender')

fig.show()
fig = px.pie(episode, names='flavour_of_tea', title='Preferred Tea')

fig.show()


fig = go.Figure(data=[go.Pie(labels=episode['heroes_location'], hole=.5)])

fig.update_layout(title="Location of Speakers")

fig.show()
fig = go.Figure(data=[go.Pie(labels=episode['heroes_nationality'], hole=.5)])

fig.update_layout(title="Nationality of Speakers")

fig.show()
fig = go.Figure(data=[go.Pie(labels=episode['recording_time'],  pull=[0, 0, 0.2, 0])])

fig.update_layout(title="Recording Time")

fig.show()
fig = px.bar(episode, x='heroes_gender', title="Gender of Speakers")

fig.show()
fig = px.bar(episode, x='category', title="Industry Category of Speakers")

fig.show()
topavgwatch=episode.nlargest(20,['youtube_avg_watch_duration'])

fig = px.bar(topavgwatch, x='heroes', y='youtube_avg_watch_duration',

             hover_data=['episode_duration', 'youtube_impression_views'], color='category',

             height=400, title="Top 20 speakers with best youtube_avg_watch_duration time")

fig.show()
#top 10 speakers based on apple listeners

apple=episode.nlargest(10,['apple_listeners'])



fig = px.line(apple, x='spotify_listeners', y='apple_listeners',color='category',title='Compare Spotify Vs Apple listeners')

fig.show()
largestviews=episode.nlargest(20,['youtube_views'])
fig = px.scatter(largestviews, x="youtube_subscribers", y="youtube_views", color="heroes",

                 size='youtube_avg_watch_duration', hover_data=['youtube_ctr'], title="Top 20 Speakers based on YouTube views")

fig.show()
largestctr=episode.nlargest(20,['youtube_ctr'])
fig = px.bar(largestctr, y='youtube_ctr', x='heroes', text='youtube_likes', title="Top 20 Speakers based on YouTube CTR")

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()


fig = px.bar(episode, x="apple_avg_listen_duration", y="heroes", orientation='h', title='Apple avg listen duration for Speakers')



fig.show()
fig = go.Figure(data=[

    go.Bar(name='Apple Listeners', x=episode['heroes'], y=episode['apple_listeners']),

    go.Bar(name='Spotify Listeners', x=episode['heroes'], y=episode['spotify_listeners']),

    

])



fig.update_layout(barmode='stack', title="Comparison of Apple and Spotify Listeners for Speakers",xaxis_tickangle=45)

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Likes', x=largestviews['heroes'], y=largestviews['youtube_likes']),

    go.Bar(name='Dislikes', x=largestviews['heroes'], y=largestviews['youtube_dislikes']),

    go.Bar(name='Comments', x=largestviews['heroes'], y=largestviews['youtube_comments']),

    go.Bar(name='Subscribers', x=largestviews['heroes'], y=largestviews['youtube_subscribers'])

])



fig.update_layout(barmode='group', title="Comparison of YouTube metrics for Top Speakers based on YouTube Views")

fig.show()
fig = px.sunburst(largestviews, path=['heroes_location', 'episode_duration', 'youtube_watch_hours'], values='youtube_views', color='heroes',title='Hierarchy of Video Duration')

fig.show()
fig = px.scatter(largestviews, x="episode_duration", y="youtube_avg_watch_duration",

        size="youtube_views", color="heroes",

                 hover_name="category", log_x=True, size_max=60, title="Compare Episode duration and YouTube avg watch duration")

fig.show()
largest5=largestviews.nlargest(5,['youtube_views'])

largest5
#WordCloud for Jeremy

epi27=pd.read_csv("/kaggle/input/chai-time-data-science/Cleaned Subtitles/E27.csv")

text27=epi27['Text']

text27 = " ".join(des for des in epi27.Text)

print ("There are {} words in the combination of all review.".format(len(text27)))

stopwords = set(STOPWORDS)

stopwords.update(["you", "know", "so", "think", "to",'of','yeah','want','people','first','And','okay','really','I'])



wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="white",stopwords = stopwords).generate(text27)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
#WordCloud for Parul

epi58=pd.read_csv("/kaggle/input/chai-time-data-science/Cleaned Subtitles/E58.csv")

text58=epi58['Text']

text58 = " ".join(des for des in epi58.Text)

print ("There are {} words in the combination of all review.".format(len(text58)))

stopwords = set(STOPWORDS)

stopwords.update(["you", "know", "so", "think", "to",'of','yeah','want','people','first','And','okay','really','I','lot','work','going'])



wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="white",stopwords = stopwords).generate(text58)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
#WordCloud for Abhishek

epi1=pd.read_csv("/kaggle/input/chai-time-data-science/Cleaned Subtitles/E1.csv")

text1=epi1['Text']

text1 = " ".join(des for des in epi1.Text)

print ("There are {} words in the combination of all review.".format(len(text1)))

stopwords = set(STOPWORDS)

stopwords.update(["you", "know", "So", "think", "to",'of','Yeah','want','people','first','And','okay','really','I'])



wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="white",stopwords = stopwords).generate(text1)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()


fig = px.line(largestviews, x='recording_date', y='heroes',color='recording_time',title="Recording Date vs Recording Time")

fig.show()
fig = px.line(largestviews, x='release_date', y='heroes',color='heroes_location', title="Location vs Release Date")

fig.show()