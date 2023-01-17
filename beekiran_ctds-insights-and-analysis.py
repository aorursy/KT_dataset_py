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
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")

%matplotlib inline
episodes = pd.read_csv('../input/chai-time-data-science/Episodes.csv')
episodes.head()

episodes.info()
yt_thumbnails = pd.read_csv('../input/chai-time-data-science/YouTube Thumbnail Types.csv')
yt_thumbnails
an_thumbnails = pd.read_csv('../input/chai-time-data-science/Anchor Thumbnail Types.csv')
an_thumbnails
desc = pd.read_csv('../input/chai-time-data-science/Description.csv')
desc.head()
import re

import string



def preprocess_two(x):

    x = x.lower()

    x = re.sub(r'@[A-Za-z0-9]+','',x)#remove usernames

    x = re.sub(r'^rt[\s]+', '', x)

    x = re.sub('https?://[A-Za-z0-9./]+','',x) # remove url

    x = re.sub(r'#([^\s]+)', r'\1', x)# remove hastags

    x = re.sub('[^a-zA-Z]', ' ', x)

    #x= re.sub(r':','',x)

    #tok = WordPunctTokenizer()

    #words = tok.tokenize(x)

    #x = (' '.join(x)).strip()

    #return [word for word in x if word not in stopwords.words('english')]

    

    clean_list.append(x)

    return x

    

    

desc_list= desc['description'].tolist()

clean_list=[]



for i in desc_list:

    preprocess_two(i)



desc['cleaned desc'] = clean_list
desc.head()
from wordcloud import WordCloud, STOPWORDS



unique_string=(" ").join(clean_list)

wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)
plt.figure(figsize=(15,4))

sns.countplot(x="flavour_of_tea",data=episodes,palette='BrBG')
plt.figure(figsize=(15,4))

plt.xticks(rotation=90)

sns.countplot(x="flavour_of_tea",data=episodes,palette='BrBG',hue='heroes_gender')
plt.figure(figsize=(15,8))

plt.legend(loc=2)

sns.countplot(x='category',data=episodes,hue='flavour_of_tea',palette='BrBG')
plt.figure(figsize=(15,8))

plt.xticks(rotation=90)

sns.countplot(x='flavour_of_tea',data=episodes,hue='recording_time',palette='BrBG')
sns.countplot(x='heroes_gender',data=episodes,palette='BrBG')
sns.countplot(x='category',data=episodes,palette='BrBG')
sns.barplot(x='recording_time',y='episode_duration',data=episodes,palette="BrBG")
sns.barplot(x='recording_time',y='episode_duration',data=episodes,palette="BrBG",hue='heroes_gender')
plt.figure(figsize=(15,8))

sns.barplot(x='recording_time',y='episode_duration',data=episodes,palette="BrBG",hue='category')
subs=episodes.sort_values(by=['youtube_subscribers'],ascending=False)

plt.figure(figsize=(15,4))

plt.tight_layout()

plt.xticks(rotation=90)

sns.barplot(x='heroes',y='youtube_subscribers',data=subs.head(10),palette='BrBG')
sns.barplot(x='category',y='youtube_views',data=episodes,palette='BrBG')
sns.barplot(x='category',y='youtube_likes',data=episodes,palette='BrBG')
sns.barplot(x='category',y='youtube_dislikes',data=episodes,palette='BrBG')
plt.figure(figsize=(15,4))

plt.xticks(rotation=90)

plt.tight_layout()

sns.lineplot(x='episode_id',y='episode_duration',data=episodes)
plt.figure(figsize=(15,4))

plt.xticks(rotation=90)

plt.tight_layout()

sns.lineplot(x='episode_id',y='episode_duration',data=episodes,hue='category')
plt.figure(figsize=(15,4))

plt.xticks(rotation=90)

plt.tight_layout()

sns.lineplot(x='episode_id',y='youtube_avg_watch_duration',data=episodes,hue='category')
plt.figure(figsize=(15,4))

plt.xticks(rotation=90)

plt.tight_layout()

sns.lineplot(x='episode_id',y='apple_avg_listen_duration',data=episodes,hue='category')
plt.figure(figsize=(15,4))

plt.xticks(rotation=90)

plt.tight_layout()

sns.lineplot(x='episode_id',y='spotify_streams',data=episodes,hue='category')
plt.figure(figsize=(8,10))

sns.barplot(y='heroes',x='anchor_plays',data = episodes.sort_values(by=['anchor_plays'],ascending=False).head(10),palette='BrBG')
plt.figure(figsize=(5,7))

sns.barplot(y='heroes',x='anchor_plays',data = episodes.sort_values(by=['anchor_plays'],ascending=True).head(10),palette='BrBG')
plt.figure(figsize=(8,12))

sns.barplot(y='heroes',x='youtube_avg_watch_duration',

            data=episodes.sort_values(by=['youtube_avg_watch_duration'],ascending=False).head(10),palette='BrBG')
plt.figure(figsize=(8,12))

sns.barplot(y='heroes',x='youtube_views',

            data=episodes.sort_values(by=['youtube_views'],ascending=False).head(10),palette='BrBG')
plt.figure(figsize=(8,12))

sns.barplot(y='heroes',x='spotify_streams',

            data=episodes.sort_values(by=['spotify_streams'],ascending=False).head(10),palette='BrBG')
plt.figure(figsize=(8,12))

sns.barplot(y='heroes',x='apple_listeners',

            data=episodes.sort_values(by=['apple_listeners'],ascending=False).head(10),palette='BrBG')
plt.figure(figsize=(15,4))

plt.tight_layout()

plt.xticks(rotation=90)

sns.countplot(x='heroes_nationality',data=episodes,palette='BrBG')
cor=episodes.groupby(['heroes_location','heroes_nationality']).count()['heroes'].unstack()
plt.figure(figsize=(20,10))

sns.heatmap(cor,cmap='YlGnBu',annot=True)