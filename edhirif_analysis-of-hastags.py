import pandas as pd

import nltk

import re

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from textblob import TextBlob



df = pd.read_csv('../input/Hurricane_Harvey.csv',encoding="ISO-8859-1")
tweets = df["Tweet"].astype("str").dropna()

tweets =''.join(tweets)

tweets = re.sub(r'[^\x00-\x7F]+',' ', tweets)

hashtags_list = re.findall('#[a-z]+',tweets)

fd = nltk.FreqDist(hashtags_list)

print(fd.most_common(10))

fd.plot(10, cumulative=False)
df_means = pd.DataFrame(columns=('hashtag','retweets','replies','likes'))



df_hurricaneharvey = df[df['Tweet'].str.contains('#hurricaneharvey',na=False)]

retweets = df_hurricaneharvey.Retweets.mean()

replies = df_hurricaneharvey.Replies.mean()

likes = df_hurricaneharvey.Likes.mean()

df_means.loc[1] = ['#hurricaneharvey',retweets,replies,likes]





df_hurricane = df[df['Tweet'].str.contains('#hurricane',na=False)]

retweets = df_hurricane.Retweets.mean()

replies = df_hurricane.Replies.mean()

likes = df_hurricane.Likes.mean()

df_means.loc[2] = ['#hurricane',retweets,replies,likes]



df_harvey = df[df['Tweet'].str.contains('#harvey',na=False)]

retweets = df_harvey.Retweets.mean()

replies = df_harvey.Replies.mean()

likes = df_harvey.Likes.mean()

df_means.loc[3] = ['#harvey',retweets,replies,likes]



df_pray = df[df['Tweet'].str.contains('#prayfortexas',na=False)]

retweets = df_pray.Retweets.mean()

replies = df_pray.Replies.mean()

likes = df_pray.Likes.mean()

df_means.loc[4] = ['#prayfortexas',retweets,replies,likes]



df_txwx = df[df['Tweet'].str.contains('#txwx',na=False)]

retweets = df_txwx.Retweets.mean()

replies = df_txwx.Replies.mean()

likes = df_txwx.Likes.mean()

df_means.loc[5] = ['#txwx',retweets,replies,likes]



df_t = df[df['Tweet'].str.contains('#t',na=False)]

retweets = df_t.Retweets.mean()

replies = df_t.Replies.mean()

likes = df_t.Likes.mean()

df_means.loc[6] = ['#t',retweets,replies,likes]



df_texas = df[df['Tweet'].str.contains('#texas',na=False)]

retweets = df_texas.Retweets.mean()

replies = df_texas.Replies.mean()

likes = df_texas.Likes.mean()

df_means.loc[7] = ['#texas',retweets,replies,likes]



df_txwxhttps = df[df['Tweet'].str.contains('#txwxhttps',na=False)]

retweets = df_txwxhttps.Retweets.mean()

replies = df_txwxhttps.Replies.mean()

likes = df_txwxhttps.Likes.mean()

df_means.loc[8] = ['#txwxhttps',retweets,replies,likes]



df_news = df[df['Tweet'].str.contains('#news',na=False)]

retweets = df_news.Retweets.mean()

replies = df_news.Replies.mean()

likes = df_news.Likes.mean()

df_means.loc[9] = ['#news',retweets,replies,likes]



df_category = df[df['Tweet'].str.contains('#category',na=False)]

retweets = df_category.Retweets.mean()

replies = df_category.Replies.mean()

likes = df_category.Likes.mean()

df_means.loc[10] = ['#category',retweets,replies,likes]



df_means.plot.barh(x=df_means.hashtag,stacked=True)

plt.show()
tweets = df_txwx["Tweet"].astype("str").dropna()

tweets =''.join(tweets)

tweets = re.sub(r'[^\x00-\x7F]+',' ', tweets)

tweets = re.sub('http[s]?:\/\/([\w]+.?[\w]+.?\/?)+','',tweets)

tweets = re.sub(r'[\W]+',' ',tweets)

tweets_txwx = tweets



tokens = word_tokenize(tweets)

tokens = [w.lower() for w in tokens if w.isalpha()]

stopwords = nltk.corpus.stopwords.words('english')

tokens = [w for w in tokens if w.lower() not in stopwords]

tokens = [w for w in tokens if len(w) > 3]



fd = nltk.FreqDist(tokens)

print('25 most common items for #txwx')

print(fd.most_common(25))
tweets = df_hurricaneharvey["Tweet"].astype("str").dropna()

tweets =''.join(tweets)

tweets = re.sub(r'[^\x00-\x7F]+',' ', tweets)

tweets = re.sub('http[s]?:\/\/([\w]+.?[\w]+.?\/?)+','',tweets)

tweets = re.sub(r'[\W]+',' ',tweets)

tweets_hurricaneharvey = tweets



tokens = word_tokenize(tweets)

tokens = [w.lower() for w in tokens if w.isalpha()]

stopwords = nltk.corpus.stopwords.words('english')

tokens = [w for w in tokens if w.lower() not in stopwords]

tokens = [w for w in tokens if len(w) > 3]



fd = nltk.FreqDist(tokens)

print('25 most common items for #hurricaneharvey')

print(fd.most_common(25))
blob_h = TextBlob(tweets_hurricaneharvey)

print(blob_h.sentiment)
blob_t = TextBlob(tweets_txwx)

print(blob_t.sentiment)