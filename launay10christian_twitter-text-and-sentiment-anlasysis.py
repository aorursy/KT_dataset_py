# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data file
tweets = pd.read_csv('../input/tweets.csv')

#Separating out the time variable by Hour, Day, Month and Year 
#for further analysis using datetime package
import datetime as dt
tweets['time'] = pd.to_datetime(tweets['time'])
tweets['hour'] = tweets['time'].apply(lambda x: x.hour)
tweets['month'] = tweets['time'].apply(lambda x: x.month)
tweets['day'] = tweets['time'].apply(lambda x: x.day)
tweets['year'] = tweets['time'].apply(lambda x: x.year)

tweets.head(5)
#Total number of tweets by both of the twitter handles
sns.countplot(x='handle', data = tweets)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,3))

#Total number of original tweets and retweets for each of the contender
retweet_hc = tweets.loc[(tweets['handle']=='HillaryClinton'), ['is_retweet']]
retweet_dt = tweets.loc[(tweets['handle']=='realDonaldTrump'), ['is_retweet']]
ax1 = sns.countplot(retweet_hc['is_retweet'], palette='rainbow', ax=axis1)
ax1.set_title("HillaryClinton's tweets")
ax1.set(xticklabels=["Tweets","Retweets"])

ax2 = sns.countplot(retweet_dt['is_retweet'], palette="Set1", ax=axis2)
ax2.set_title("realDonaldTrump's tweets")
ax2.set(xticklabels=["Tweets","Retweets"])
#Number of tweets by the months
monthly_tweets = tweets.groupby(['month', 'handle']).size().unstack()
monthly_tweets.plot(title='Monthly Tweet Counts', colormap='winter')
#Number of tweets daily
daily_tweets = tweets.groupby(['day', 'handle']).size().unstack()
daily_tweets.plot(title='Daily Tweet Counts')
#Number of tweets hourly
hourly_tweets = tweets.groupby(['hour', 'handle']).size().unstack()
hourly_tweets.plot(title='Hourly Tweet Counts', colormap='coolwarm')
from wordcloud import WordCloud
from wordcloud import STOPWORDS

tweets_hillary=tweets.loc[(tweets['handle']=='HillaryClinton'),['text']]
tweets_trump=tweets.loc[(tweets['handle']=='realDonaldTrump'),['text']]
stopwords = set(STOPWORDS)
stopwords.add("http")
stopwords.add("https")
stopwords.add("amp")
stopwords.add("CO")
stopwords.add("Trump")
stopwords.add("Trump2016")
stopwords.add("Donald")
stopwords.add("Clinton")
stopwords.add("Hillary")
stopwords.add("realDonaldTrump")
stopwords.add("will")
stopwords.add("say")
stopwords.add("said")
stopwords.add("let")
stopwords.add("vote")
stopwords.add("now")
stopwords.add("go")
wordcloud_hc = WordCloud(background_color='white',max_font_size=46, relative_scaling=0.5,stopwords=stopwords).generate(tweets_hillary['text'].str.cat())
plt.figure()
plt.imshow(wordcloud_hc)
plt.axis("off")
plt.show()
wordcloud_dt = WordCloud(max_font_size=42, relative_scaling=.5,stopwords=stopwords).generate(tweets_trump['text'].str.cat())
plt.figure()
plt.imshow(wordcloud_dt)
plt.axis("off")
plt.show()
from textblob import TextBlob

bloblist_desc = list()

df_tweet_descr_str=tweets['text'].astype(str)
for row in df_tweet_descr_str:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    df_tweet_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])
 
def f(df_tweet_polarity_desc):
    if df_tweet_polarity_desc['sentiment'] > 0:
        val = "Positive"
    elif df_tweet_polarity_desc['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

df_tweet_polarity_desc['Sentiment_Type'] = df_tweet_polarity_desc.apply(f, axis=1)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=df_tweet_polarity_desc)
tweet_clinton = tweets.loc[(tweets['handle']=='HillaryClinton'), ['text']]
bloblist_desc = list()

df_tweet_clinton_str=tweet_clinton['text'].astype(str)
for row in df_tweet_clinton_str:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    df_tweet_clinton_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])
 
def f(df_tweet_clinton_polarity_desc):
    if df_tweet_clinton_polarity_desc['sentiment'] > 0:
        val = "Positive"
    elif df_tweet_clinton_polarity_desc['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

df_tweet_clinton_polarity_desc['Sentiment_Type'] = df_tweet_clinton_polarity_desc.apply(f, axis=1)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=df_tweet_clinton_polarity_desc)
          

tweet_trump = tweets.loc[(tweets['handle']=='realDonaldTrump'), ['text']]
bloblist_desc = list()

df_tweet_trump_str=tweet_trump['text'].astype(str)
for row in df_tweet_trump_str:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    df_tweet_trump_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])
 
def f(df_tweet_trump_polarity_desc):
    if df_tweet_trump_polarity_desc['sentiment'] > 0:
        val = "Positive"
    elif df_tweet_trump_polarity_desc['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

df_tweet_trump_polarity_desc['Sentiment_Type'] = df_tweet_trump_polarity_desc.apply(f, axis=1)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=df_tweet_trump_polarity_desc)