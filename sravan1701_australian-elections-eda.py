# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

print(os.listdir("../input"))



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS 

from nltk.corpus import stopwords 

from nltk.stem.wordnet import WordNetLemmatizer

from textblob import TextBlob

import string

from collections import Counter

import folium

# Any results you write to the current directory are saved as output.
auspol_df=pd.read_csv('../input/auspol2019.csv')

auspol_df.head()
#Loading the geocode data

loc_geo_df=pd.read_csv('../input/location_geocode.csv')

loc_geo_df.head()
#combine both the dataset on the name column

tweets_df=auspol_df.merge(loc_geo_df,how='inner',left_on='user_location',right_on='name')

tweets_df.head()
#checking the data types

tweets_df.isnull().sum()
#drop the missing values 

tweets_df=tweets_df.dropna()
#Drop the unneccessary columns whihch are not helpful in the analysis

tweets_df=tweets_df.drop(['id','user_id','user_screen_name',"user_created_at","name"],axis=1)

tweets_df.head()
#lets explore created_at column

tweets_df['created_at'] =  pd.to_datetime(tweets_df['created_at'])

cnt_srs = tweets_df['created_at'].dt.date.value_counts()

cnt_srs = cnt_srs.sort_index()

plt.figure(figsize=(14,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.7, color='Blue')

plt.xticks(rotation='vertical')

plt.xlabel('Date', fontsize=12)

plt.ylabel('Number of tweets', fontsize=10)

plt.title("Number of tweets according to dates")

plt.show()
tweets_df['created_at'] =  pd.to_datetime(tweets_df['created_at'])

tweets_df_day=tweets_df.loc[(tweets_df['created_at'].dt.month== 5) & (tweets_df['created_at'].dt.day==18), 'created_at']

cnt_srs_hour=tweets_df_day.dt.hour.value_counts()

plt.figure(figsize=(14,6))

sns.barplot(cnt_srs_hour.index, cnt_srs_hour.values, alpha=0.7, color='Blue')

plt.xticks(rotation='vertical')

plt.xlabel('Hour', fontsize=12)

plt.ylabel('Number of tweets', fontsize=10)

plt.title("Number of tweets hour by hour on 18th May")

plt.show()
#Daywise Distribution

tweets_df['created_at'] =  pd.to_datetime(tweets_df['created_at'])

cnt_srs = tweets_df['created_at'].dt.weekday_name.value_counts()

cnt_srs = cnt_srs.sort_index()

plt.figure(figsize=(14,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.7, color='Blue')

plt.xticks(rotation='vertical')

plt.xlabel('Date', fontsize=12)

plt.ylabel('Number of tweets', fontsize=10)

plt.title("Number of tweets according to day")

plt.show()
#most favourite and retweeted tweet

print(f" Maximum number of retweets {tweets_df.retweet_count.max()}")

print(f" Maximum number of favorites {tweets_df.favorite_count.max()}")
#checking the tweet which has the maximum retweet count

tweets_df.loc[tweets_df['retweet_count']==6622.0,['full_text','user_name']].values
#lets see the tweet which has the maximum retweet count

tweets_df.loc[tweets_df['favorite_count']==15559.0,['full_text','user_name']].values
#Top tweeters along with user names

tweets_df.user_name.value_counts()[:10,]
plt.figure(figsize=(14,7))

cnt_user_location = tweets_df['user_location'].value_counts()

cnt_user_location.reset_index()

cnt_user_location = cnt_user_location[:10,]

sns.barplot(cnt_user_location.index, cnt_user_location.values,data=tweets_df)

plt.xticks(rotation='vertical')

plt.show()
#using the latitudes and longtiudes to determine the exact location of the user

tweets_df['count'] = 1

tweets_df[['lat', 'long', 'count']].groupby(['lat', 'long']).sum().sort_values('count', ascending=False).head(10)
def generateBaseMap(default_location=[-25.734968,134.489563], default_zoom_start=4):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    return base_map
from folium.plugins import HeatMap

base_map = generateBaseMap()

HeatMap(data=tweets_df[['lat', 'long', 'count']].groupby(['lat', 'long']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)

base_map
stop = set(stopwords.words('english'))

exclude = set(string.punctuation) 

lemma = WordNetLemmatizer()

data_samples=tweets_df['user_description']

def clean(doc):

    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])

    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)

    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

    return normalized



doc_clean = [clean(doc).split() for doc in data_samples]  

norm_doc = [clean(doc) for doc in data_samples]
plt.figure(figsize=(14,6))

wordcloud = WordCloud().generate(str(norm_doc))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
#wword cloud on the full text

data_samples1=tweets_df['full_text']

doc_clean = [clean(doc).split() for doc in data_samples1]  

norm_doc = [clean(doc) for doc in data_samples1]

plt.figure(figsize=(14,6))

wordcloud = WordCloud().generate(str(norm_doc))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
# from tetxblob import TextBlob

tweets_df['sentiment'] = data_samples1.apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
#mapping the polarity values to different labels

tweets_df['Polarity']=pd.cut(tweets_df['sentiment'],[-np.inf, -.01, .01, np.inf],

                             labels=['Negative','Neutral','Positive'])

tweets_df[['sentiment','Polarity']].head(10)
norm_corpus_df = pd.DataFrame({'Document': data_samples1, 

                          'Category': tweets_df['Polarity']})

norm_corpus_df = norm_corpus_df[['Document', 'Category']]

norm_corpus_df.head()