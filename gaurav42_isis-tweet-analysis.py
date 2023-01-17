import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
df = pd.read_csv('../input/tweets.csv', header=0)
print(len(df))
unique_tweeterites = df.groupby('username')
print(len(unique_tweeterites))
df['date'] = df['time'].map(lambda x: pd.to_datetime(str(x).split()[0]))
tweets_by_day = df.groupby('date').count().reset_index()
len(tweets_by_day)
tweets_by_day = tweets_by_day.sort_values(by='date')
x = tweets_by_day['date']
y = tweets_by_day['name']
plt.xlabel('Date')
plt.ylabel('Number of tweets')
plt.xticks(rotation=45)
plt.title('Number of tweets trend by dates')
plt.plot(x, y, label='Tweets trend by days')
plt.show()
df['date'].hist(bins=8)
plt.xticks(rotation=45)
plt.show()
top_10_max_tweets_days = tweets_by_day.sort_values(by='username').tail(10)
x = top_10_max_tweets_days['date']
y = top_10_max_tweets_days['name']
plt.xlabel('Date')
plt.ylabel('Number of tweets')
plt.title('Most number of tweets in a day')
plt.xticks(range(10), x, rotation=45)
plt.bar(range(10), y, label='Most tweets in a day')
plt.show()
tweeterites = df.groupby(['username']).count().reset_index()
tweeterites = tweeterites.sort_values(by='tweets').tail(10)
x = tweeterites['username']
y = tweeterites['tweets']
plt.xlabel('Twitter handle')
plt.ylabel('Number of tweets')
plt.title('Most number of tweets by user')
plt.xticks(range(10), x, rotation=45)
plt.bar(range(10), y, label='Most tweets+retweets by user')
plt.show()
most_followed_users = df.drop_duplicates('username', keep='last')
most_followed_users_top_10 = most_followed_users.sort_values(by='followers').tail(10)
x = most_followed_users_top_10['username']
y = most_followed_users_top_10['followers']
plt.xlabel('Username')
plt.ylabel('Followers')
plt.title('Most followed user')
plt.xticks(range(10), x, rotation=60)
plt.bar(range(10), y, label='Most followed user')
plt.show()
MyColumns = ['hashtag','cnt']
hashtagcount = pd.DataFrame(columns=MyColumns)

for index, row in df.iterrows():
    if '#' in row['tweets']:
        words = row['tweets'].split()
        for word in words:
            if word[0] == '#':
                hashtagcount.loc[len(hashtagcount)] = [word, 1]  # adding a row
                
hashtags = hashtagcount.groupby(['hashtag']).count().reset_index()
hashtags = hashtags.sort_values(by='cnt').tail(10)

x = hashtags['hashtag']
y = hashtags['cnt']
plt.xlabel('hashtag')
plt.ylabel('Number of times used')
plt.title('Most number of hashtags used')
plt.xticks(range(10), x, rotation=60)
plt.bar(range(10), y, label='Most hashtags used')
plt.show()
