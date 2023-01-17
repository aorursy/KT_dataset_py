!pip install dexplot
import pandas as pd
import numpy as np
import dexplot as dxp
import plotly.express as px
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image

#plotly
!pip install chart_studio
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


import re                                  # library for regular expression operations
import string                              # for string operations
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings

#datetime
from datetime import datetime
import os


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


ctds_episodes = pd.read_csv('../input/chai-time-data-science/Episodes.csv',parse_dates=['recording_date','release_date'])
youtube_thumbnails = pd.read_csv('../input/chai-time-data-science/YouTube Thumbnail Types.csv')
anchor_thumbnails = pd.read_csv('../input/chai-time-data-science/Anchor Thumbnail Types.csv')
description = pd.read_csv('../input/chai-time-data-science/Description.csv')


print("youtube_thumbnails dataset")
youtube_thumbnails.head()

print("anchor_thumbnails dataset")
anchor_thumbnails.head()
print("youtube_thumbnails dataset")
youtube_thumbnails.head()

print("CTDS Episodes Dataset")
ctds_episodes.head()

ctds_episodes.info()
labels = ctds_episodes['heroes_gender'].value_counts()[:10].index
values = ctds_episodes['heroes_gender'].value_counts()[:10].values
colors=['#2678bf',
 '#98adbf']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial',marker=dict(colors=colors))])
fig.show()
labels = ctds_episodes['heroes_nationality'].value_counts()[:10].index
values = ctds_episodes['heroes_nationality'].value_counts()[:10].values
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

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial',marker=dict(colors=colors))])
fig.show()
dxp.count(val='category', data=ctds_episodes,cmap='tab10',figsize=(4,3),normalize=True)

dxp.count(val='category', data=ctds_episodes,normalize=True,split='heroes_gender',figsize=(4,3))
dxp.count(val='heroes_nationality', data=ctds_episodes, split='category',normalize=True,figsize=(10,6),size=0.9,stacked=True)
dxp.count(val='flavour_of_tea', data=ctds_episodes,normalize=True,figsize=(6,3))
ctds_episodes['episode_id'].count()
labels = ctds_episodes['recording_time'].value_counts()[:10].index
values = ctds_episodes['recording_time'].value_counts()[:10].values
colors=['#bfbfbf','#3795bf','#2678bf','#98adbf']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',
                             insidetextorientation='radial',marker=dict(colors=colors))])
fig.show()
dxp.count(val='flavour_of_tea', data=ctds_episodes, split='recording_time', 
          orientation='v', stacked=True)
ctds_episodes['episode_duration'].iplot(kind='area',fill=True,opacity=1,xTitle='Episode',yTitle='Duration(sec)')
df = ctds_episodes[['release_date','episode_duration']]
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

font = '../input/quicksandboldttf/Quicksand-Bold.ttf'
word_cloud = WordCloud(font_path=font,
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
