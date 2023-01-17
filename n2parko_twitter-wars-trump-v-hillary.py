# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('fivethirtyeight')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/tweets.csv')
# remove timestamp & group by 

df['time'] = pd.to_datetime(df.time)

df['date']= df.time.apply(lambda x: x.date())

df['week']= df.time.apply(lambda x: x.isocalendar()[1])

df['tweet_hour'] = df.time.apply(lambda x: (x + pd.Timedelta(hours=-5)).hour) # convert utc to est



# filter retweets and dates after 4/18/2016

df_ex_rt = df[(df['is_retweet'] == False) & (df['date'] >= pd.to_datetime('2016-04-18').date()) & (df['date'] < pd.to_datetime('2016-09-28').date())]

df_ex_rt_daily = df_ex_rt.groupby(['date', 'handle']).size().unstack()

df_ex_rt_weekly = df_ex_rt.groupby(['week', 'handle']).size().unstack()



# plot the timeseries

df_ex_rt_daily.tail(20).plot(kind='bar', title='Daily Tweet Counts')

df_ex_rt_weekly.plot(kind='bar', title='Weekly Tweet Counts')



df_ex_rt_daily.describe()
# when do candidates tweet

bins = np.linspace(0, 24, 24)

plt.hist(df_ex_rt['tweet_hour'][df_ex_rt['handle']=='HillaryClinton'], bins, alpha=0.6, label="Hillary")

plt.hist(df_ex_rt['tweet_hour'][df_ex_rt['handle']=='realDonaldTrump'], bins, alpha=0.6, label="Donald")

plt.legend()

plt.title('Tweet Counts by Hour')

plt.show()
# does time of tweet correlate with likes

df_hour_retweets = df_ex_rt.groupby(['tweet_hour', 'handle']).apply(lambda x: np.mean(x.retweet_count)).unstack()



df_hour_retweets.plot(title='Avg Retweets by Hour Tweeted')
# does time of tweet correlate with likes

df_hour_likes = df_ex_rt.groupby(['tweet_hour', 'handle']).apply(lambda x: np.mean(x.favorite_count)).unstack()

df_hour_likes.plot(title='Avg Likes by Hour Tweeted')
df_ex_rt.groupby(['handle']).describe()