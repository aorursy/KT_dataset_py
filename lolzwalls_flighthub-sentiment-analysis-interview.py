# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as datetime

import calendar



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read CSV's in DFs

df_sent = pd.read_csv('/kaggle/input/flighthub-twitter-analysis/flighthub_tweets_sentiment.csv', index_col='date', parse_dates=True)

df_ent = pd.read_csv('/kaggle/input/flighthub-twitter-analysis/flighthub_tweets_entities.csv')

# Drop unnecessary columns

df_sent.drop(columns=['Unnamed: 0'], inplace=True)

df_sent.head(3)
df_sent.describe()
plt.figure(figsize=(10,6))

plt.title("Distribution of Customer Sentiment via Tweets")

#sns.kdeplot(data=df_sent['sentiment_score'], shade=True, clip=(-1.0, 1.0))

sns.distplot(a=df_sent['sentiment_score'], kde=False)

plt.xlim(-1.0,1.0)

plt.xlabel("Sentiment Score")
plt.figure(figsize=(10,6))

sns.jointplot(x=df_sent['sentiment_score'], y=df_sent['sentiment_magnitude'], kind='kde')

plt.ylabel("Sentiment Magnitude")

plt.xlabel("Sentiment Score")
plt.figure(figsize=(14,6))

plt.title("Flighthub Customer Sentiment Over Time")

sns.lineplot(data=df_sent['sentiment_score'], label='Customer Sentiment')

plt.xlabel("Date")

plt.ylabel("Sentiment Score")

plt.ylim(-1.0, 1.0)
date_YM = df_sent.index

date_YM = [str(d.year) + '-' + str(d.month) for d in date_YM]

date_YM = [datetime.datetime.strptime(d, '%Y-%m').date() for d in date_YM]
df_volume = df_sent.groupby([df_sent.index.year.values, df_sent.index.month.values]).transform('count')

df_volume.index = date_YM



plt.figure(figsize=(14,6))

plt.title("Volume of Tweets Mentioning Flighthub by Month")

sns.barplot(x=df_volume.index, y=df_volume['content'])

plt.ylabel("Volume of Tweets")

plt.xlabel("Month")
df_monthly = df_sent.groupby([df_sent.index.year.values,df_sent.index.month.values]).mean()

date_YM = df_monthly.index

date_YM[0][1]

date_YM = [str(d[0]) + '-' + str(d[1]) for d in date_YM]

date_YM = [datetime.datetime.strptime(d, '%Y-%m').date() for d in date_YM]

df_monthly.index = date_YM
plt.figure(figsize=(14,6))

plt.title("Average Flighthub Customer Sentiment by Month")

sns.barplot(x=df_monthly.index, y=df_monthly['sentiment_score'])

plt.ylabel("Sentiment Score")

plt.xlabel("Month")