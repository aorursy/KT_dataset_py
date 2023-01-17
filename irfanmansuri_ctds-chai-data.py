! pip install dexplot
# Importing the necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objs as go
from plotly.offline import iplot
import dexplot as dxp
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
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
youtube_TN = pd.read_csv('/kaggle/input/chai-time-data-science/YouTube Thumbnail Types.csv')
episodes = pd.read_csv('/kaggle/input/chai-time-data-science/Episodes.csv')
anchor_TN = pd.read_csv('/kaggle/input/chai-time-data-science/Anchor Thumbnail Types.csv')
results = pd.read_csv('/kaggle/input/chai-time-data-science/Results.csv')
description = pd.read_csv('/kaggle/input/chai-time-data-science/Description.csv')
print("Youtube Thumbnails")
youtube_TN.head()
print("Anchor Thumbnails")
anchor_TN
print('episodes')
episodes.head()
print("results")
results.head()
print("description")
description.head()
youtube_TN.info()
episodes.info()
anchor_TN.info()
results.info()
description.info()
# integrate episodes and thumbnail types datasets
import datetime as datetime


episodes_youtube_TN = episodes.merge(
youtube_TN, left_on="youtube_thumbnail_type", right_on="youtube_thumbnail_type")
episodes_youtube_TN['recording_date'] = pd.to_datetime(episodes_youtube_TN["recording_date"])
episodes_youtube_TN["release_date"] = pd.to_datetime(episodes_youtube_TN["release_date"])
# Visualize the change in thumbnail types over the episodes

dxp.count(
val="release_date",
data=episodes_youtube_TN,
split="description",
orientation="h",
stacked=True,
figsize=(12,24),
xlabel="Number of episodes")
labels = episodes["heroes_gender"].value_counts()[:10].index
values = episodes["heroes_gender"].value_counts()[:10].values

colors=['#2678bf', '#98adbf']

fig = go.Figure(data=[go.Pie(labels = labels, values=values, textinfo="label+percent",
                            insidetextorientation="radial", marker=dict(colors=colors))])

fig.show()
labels = episodes["heroes_nationality"].value_counts()[:10].index
values = episodes["heroes_nationality"].value_counts()[:10].values

colors=['#bfbfbf',
 '#98adbf',
 '#1d4466',
 '#2678bf',
 '#2c6699',
 '#3780bf',
 '#3a88cc',
 '#4c89bf',
 '#729bbf',
 '#98adbf',
 '#bfbfbf'] 

fig = go.Figure(data=[go.Pie(labels=labels, values = values, textinfo="label+percent",
                            insidetextorientation="radial", marker = dict(colors=colors))])
fig.show()
labels = episodes["heroes_location"].value_counts()[:10].index
values = episodes["heroes_location"].value_counts()[:10].values

colors = episodes["heroes_location"]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent",
                             insidetextorientation="radial", marker=dict(colors=colors))])
fig.show()    
labels = episodes['flavour_of_tea'].value_counts()[:10].index
values = episodes['flavour_of_tea'].value_counts()[:10].values

colors = episodes['flavour_of_tea']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent",
                             insidetextorientation="radial", marker=dict(colors=colors))])
fig.show()    
labels = episodes['youtube_thumbnail_type'].value_counts()[:10].index
values = episodes['youtube_thumbnail_type'].value_counts()[:10].values

colors = episodes['youtube_thumbnail_type']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo="label+percent",
                             insidetextorientation="radial", marker=dict(colors=colors))])
fig.show()    
dxp.count(val='category', data=episodes, figsize=(4,3), normalize=True)
dxp.count(val='category', data=episodes, normalize=True, split="heroes_gender", figsize=(4,3))
dxp.count(val='category', data=episodes, split="heroes_nationality", figsize=(4,3), normalize=True)
dxp.count(val="heroes_nationality", data=episodes, split="heroes_gender", normalize=True, figsize=(10,6),
         size=0.9, stacked=True)
dxp.count(val="heroes_nationality", data=episodes, split="category", normalize=True, stacked=True, figsize=(10,6), size=0.9)
dxp.count(val="heroes_nationality", data=episodes, split="youtube_dislikes", figsize=(10,3), normalize=True)
dxp.count(val="heroes_nationality", data=episodes, split="youtube_comments", figsize=(10,3), normalize=True)
dxp.count(val="flavour_of_tea", data=episodes, normalize=True, figsize=(8,5))
dxp.count(val="flavour_of_tea", data=episodes, split="heroes_gender", figsize=(6,4), normalize=True)
dxp.count(val="flavour_of_tea", data=episodes, split="heroes_nationality", figsize=(10,8), normalize=True,stacked=True)
episodes['episode_id'].count()
labels = episodes['recording_time'].value_counts()[:10].index
values = episodes['recording_time'].value_counts()[:10].values

colors = episodes['recording_time']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', 
                             insidetextorientation='radial', marker=dict(colors=colors))])
fig.show()
labels = episodes['recording_date'].value_counts()[:10].index
values = episodes['recording_date'].value_counts()[:10].values

colors = episodes['recording_date']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', 
                             insidetextorientation='radial', marker=dict(colors=colors))])
fig.show()
labels = episodes['release_date'].value_counts()[:10].index
values = episodes['release_date'].value_counts()[:10].values

colors = episodes['release_date']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', 
                             insidetextorientation='radial', marker=dict(colors=colors))])
fig.show()
labels = episodes['episode_duration'].value_counts()[:10].index
values = episodes['episode_duration'].value_counts()[:10].values

colors = episodes['episode_duration']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', 
                             insidetextorientation='radial', marker=dict(colors=colors))])
fig.show()
dxp.count(val='flavour_of_tea', data=episodes, split="recording_time", orientation='v', stacked=True)
dxp.count(val='flavour_of_tea', data=episodes, split="recording_date", orientation='v', figsize = (10,8), stacked=True)
dxp.count(val='flavour_of_tea', data=episodes, split="release_date", orientation='v', figsize = (10,8), stacked=True)
dxp.count(val='flavour_of_tea', data=episodes, split="episode_duration", orientation='v', figsize = (10,8), stacked=True)
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
episodes['episode_duration'].iplot(kind='area',fill=True,opacity=1,xTitle='Episode',yTitle='Duration(sec)')
df = episodes[['release_date','episode_duration']]
df.set_index('release_date').iplot(kind='scatter',mode='markers',symbol='cross',xTitle='Release Date',yTitle='Duration(sec)')
description.head()
# text preprocessing helper functions

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(tokenized_text)
    return combined_text
# Applying the cleaning function to both test and training datasets
description['description'] = description['description'].apply(str).apply(lambda x: text_preprocessing(x))
description.head()

from wordcloud import WordCloud


word_cloud = WordCloud(
                       width=1600,
                       height=800,
                       colormap='PuRd', 
                       margin=0,
                       max_words=500, # Maximum numbers of words we want to see 
                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud
                       max_font_size=150, min_font_size=20,  # Font size range
                       background_color="white").generate(" ".join(description['description']))

plt.figure(figsize=(10, 16))
plt.imshow(word_cloud, interpolation="gaussian")
plt.axis("off")
plt.show()
