# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Input data file

df = pd.read_csv('../input/tweets.csv')

df.head()
df.info()
df.describe()
#Separating out the time variable by Hour, Day, Month and Year 

#for further analysis using datetime package

import datetime as dt

df['time'] = pd.to_datetime(df['time'])

df['hour'] = df['time'].apply(lambda x: x.hour)

df['month'] = df['time'].apply(lambda x: x.month)

df['day'] = df['time'].apply(lambda x: x.day)

df['year'] = df['time'].apply(lambda x: x.year)

df.head()
#Total number of tweets by both of the twitter handles

sns.countplot(x='handle', data = df)
#Total number of original tweets and retweets for each of the contender

retweet_hc = df.loc[(df['handle']=='HillaryClinton'), ['is_retweet']]

retweet_dt = df.loc[(df['handle']=='realDonaldTrump'), ['is_retweet']]

ax1 = sns.countplot(retweet_hc['is_retweet'], palette='rainbow')

ax1.set_title("HillaryClinton's tweets")

ax1.set(xticklabels=["Tweets","Retweets"])

ax2 = sns.countplot(retweet_dt['is_retweet'], palette='rainbow')

ax2.set_title("realDonaldTrump's tweets")

ax2.set(xticklabels=["Tweets","Retweets"])
#Number of tweets by the months

monthly_tweets = df.groupby(['month', 'handle']).size().unstack()

monthly_tweets.plot(title='Monthly Tweet Counts', colormap='winter')
#Number of tweets daily

daily_tweets = df.groupby(['day', 'handle']).size().unstack()

daily_tweets.plot(title='Daily Tweet Counts', colormap='spring')
#Number of tweets hourly

hourly_tweets = df.groupby(['hour', 'handle']).size().unstack()

hourly_tweets.plot(title='Hourly Tweet Counts', colormap='coolwarm')