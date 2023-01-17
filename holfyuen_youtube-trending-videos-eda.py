import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime
# Load all trending data by country and combine it in a single dataframe



countries = ['BR','CA','IN','KR','DE','US','GB','JP','FR','MX','RU']

df = pd.DataFrame()

df1 = pd.DataFrame()



# Loop over the countries to get the whole dataset

for c in countries:

    path = '/kaggle/input/youtube-trending-video-dataset/' + c + '_youtube_trending_data.csv'

    df1 = pd.read_csv(path, parse_dates=['publishedAt','trending_date'])

    df1['country'] = c

    df = pd.concat([df, df1])



df.shape
df.head()
# Remove unused columns

df = df.drop(['thumbnail_link'], axis=1)

# trending_date column only needs date format

df.trending_date = df.trending_date.dt.date



start_date = df.trending_date.min()

end_date = df.trending_date.max()

print ('Date covered from %s to %s' % (start_date, end_date))

print ('No. of days covered: %d' % df.trending_date.nunique())
# Make list of videos

v_col = ['video_id', 'title', 'publishedAt', 'channelId', 'channelTitle']

videos = df[v_col].drop_duplicates(subset='video_id', keep='last') # Treat video_id as unique



# Compute video-based statistics

video_stat = df.groupby('video_id').agg({'video_id':'count',

    'trending_date': ['nunique','min','max'],

                           'view_count': 'max',

                           'likes': 'max',

                           'dislikes': 'max',

                           'comment_count': 'max',

                           'country': ['unique','nunique']})



video_stat.columns = ['trending_count','days_trend', 'first_trend_date','last_trend_date','views','likes','dislikes',

                      'comments','country_list','country_count']

video_stat.reset_index(inplace=True)

video_stat.head()
videos = videos.merge(video_stat, on='video_id')
# Top 10 videos that are trending the most time

videos.sort_values('trending_count', ascending=False).head(10)[['title','channelTitle','trending_count','country_list']]
# 10 Highest view videos

videos.sort_values('views', ascending=False).head(10)[['title','channelTitle','views','country_list']]
# Top 10 most liked videos

videos.sort_values('likes', ascending=False).head(10)[['title','channelTitle','likes','country_list']]
# Top 10 most dislike videos

videos.sort_values('dislikes', ascending=False).head(10)[['title','channelTitle','dislikes','country_list']]
# Top 10 videos by comments

videos.sort_values('comments', ascending=False).head(10)[['title','channelTitle','comments','country_list']]
# Trending videos with lowest views

videos.sort_values('views').head(10)[['title','channelTitle','views','country_list']]
# Oldest videos that become trending?

videos.sort_values('publishedAt').head(10)[['title','channelTitle','publishedAt','first_trend_date','country_list']]
channel_stat = videos.groupby('channelId').agg({'video_id':'count',

                               'views': ['sum','mean'],

                           'days_trend': 'sum'})

channel_stat.columns = ['no of videos','total views', 'average views', 'total days trending']

channel_stat.reset_index(inplace=True)



channel_names = df[['channelId','channelTitle']].drop_duplicates(subset='channelId', keep='last')

channel_stat = channel_stat.merge(channel_names, on='channelId')
# Channels with the greatest number of trending videos

channel_stat.sort_values('no of videos', ascending=False).head(10)[['channelTitle','no of videos']]
# Channels with highest total views

channel_stat.sort_values('total views', ascending=False).head(10)[['channelTitle','total views', 'no of videos']]
channel_stat['average views'] = channel_stat['average views'].astype('int64')
# Channels with highest average views

channel_stat.sort_values('average views', ascending=False).head(10)[['channelTitle','average views', 'no of videos']]
# Channels with largest combined days of trending

channel_stat.sort_values('total days trending', ascending=False).head(10)[['channelTitle','total days trending', 'no of videos']]
# Average view of trending video per country

df.groupby('country')['view_count'].mean().astype('int64').sort_values(ascending=False)
# Average 'hurdle' (minimum view) of trending video per country

hurdles = df.groupby(['country','trending_date'])['view_count'].min().reset_index()

hurdles.groupby('country')['view_count'].mean().astype('int64').sort_values(ascending=False)