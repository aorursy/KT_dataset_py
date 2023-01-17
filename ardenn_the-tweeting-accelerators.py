import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS
tweets=pd.read_csv("../input/the-incubator-tweets/tweets.csv",parse_dates=["created_at"])
tweets.head()
tweets["username"]=tweets["username"].astype("category")
tweets["tweet_id"].unique().all()
tweets.dtypes
#summary of the tweets

tweets.describe().astype(int)
#Find the organisation with the most tweets

tweets_per_user=tweets.groupby("username").size().reset_index()

tweets_per_user.columns=["Organisation","Tweets"]

tweets_per_user.sort_values(by="Tweets",ascending=False).iloc[0]
#Find the tweet with most retweets

retweets_per_tweet=tweets.groupby("tweet_id")["retweets"].sum().reset_index()

retweets_per_tweet.sort_values(by="retweets",ascending=False).iloc[0]
#Find the most used words

corpus = ' '.join(tweets['tweet '])

wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)

plt.figure(figsize=(12,15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
#Distribution of tweets by weekday

tweets["wkday"]=tweets["created_at"].dt.dayofweek

tweets_by_weekday=tweets.groupby("wkday").size().reset_index()

tweets_by_weekday.columns=["wkday","tweets"]

ax=sns.barplot(x="wkday",y="tweets",data=tweets_by_weekday)

ax.figure.set_size_inches(10,6)
#Distribution of tweets per month

tweets["month"]=tweets["created_at"].dt.month

tweets_per_month=tweets.groupby("month").size().reset_index()

tweets_per_month.columns=["month","tweets"]

ax=sns.barplot(x="month",y="tweets", data=tweets_per_month)

ax.figure.set_size_inches(10,6)
#Relationship btween number of retweets and the time of day/day itself

tweets["date"]=tweets["created_at"].dt.date

tweets["hour"]=tweets["created_at"].dt.hour

tweets_per_day_time=tweets.groupby(["date","hour"])["retweets"].sum().unstack("hour",fill_value=0)
hour_retweets=tweets_per_day_time.sum(axis=0).reset_index()

hour_retweets.columns=["hour","retweets"]

hour_retweets.plot(figsize=(10,6))
ax=sns.barplot(x="hour",y="retweets",data=hour_retweets)

ax.figure.set_size_inches(10,6)