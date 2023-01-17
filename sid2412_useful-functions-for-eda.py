import numpy as np 

import pandas as pd

import json



import seaborn as sns

sns.set_style('whitegrid')

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings(action="ignore")
df = pd.read_csv('../input/youtube-new/USvideos.csv')
df.head()
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df['trending_date'].head()
df['publish_time'] = pd.to_datetime(df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
df['publish_time'].head()
df.insert(5, 'publish_date', df['publish_time'].dt.date)
df['publish_time'] = df['publish_time'].dt.time
df['publish_date'] = pd.to_datetime(df['publish_date'])
id_to_cat = {}



with open('../input/youtube-new/US_category_id.json', 'r') as f:

    data = json.load(f)

    for category in data['items']:

        id_to_cat[category['id']] = category['snippet']['title']
id_to_cat
df['category_id'] = df['category_id'].astype(str)
df.insert(5, 'category', df['category_id'].map(id_to_cat))
df['video_id'].nunique()
len(df['video_id'])
print(df.shape)

df_last = df.drop_duplicates(subset=['video_id'], keep='last', inplace=False)

df_first = df.drop_duplicates(subset=['video_id'], keep='first', inplace=False)

print(df_last.shape)

print(df_first.shape)
print(df['video_id'].duplicated().any())

print(df_last['video_id'].duplicated().any())

print(df_first['video_id'].duplicated().any())
df_last.head()
def top_10(df, col, num=10):

    sort_df = df.sort_values(col, ascending=False).iloc[:num]

    

    ax = sort_df[col].plot.bar()

   

    labels = []

    for item in sort_df['title']:

        labels.append(item[:10] + '...')

        

    ax.set_title(col.upper(), fontsize=16)

    ax.set_xticklabels(labels, rotation=45, fontsize=10)

    

    return sort_df[['video_id', 'title', 'channel_title', col]]
top_10(df_last, 'views', 10)
top_10(df_last, 'comment_count')
def bottom_10(df, col, num=10):

    sort_df = df.sort_values(col, ascending=True).iloc[:num]

    

    ax1 = sort_df[col].plot.bar()

    

    labels = []

    for item in sort_df['title']:

        labels.append(item[:10] + '...')

        

    ax1.set_title('Bottom {} {} for videos'.format(num, col))

    ax1.set_xticklabels(labels, rotation=45)

    

    return sort_df[['title', 'channel_title', col]]
bottom_10(df_last, 'views')
def channel_stats(df, channel, num=5, arrange_by='views'):

    target_df = df.loc[df['channel_title'] == channel].sort_values(arrange_by, ascending=False)[:num]

    

    ax1 = target_df[['views']].plot.bar()

    

    ax2 = target_df[['likes', 'dislikes', 'comment_count']].plot.bar()

    

    labels = []

    for item in target_df['title']:

        labels.append(item[:15] + '...')

    

    ax1.set_title('Top {} views for channel {} arranged by {}'.format(num, channel, arrange_by))

    ax1.set_xticklabels(labels, rotation=45)

    

    ax2.set_title('Top {} Likes/Dislikes/Comments for channel {} arranged by {}'.format(num, channel, arrange_by))

    ax2.set_xticklabels(labels, rotation=45)

    

    return df.loc[df['channel_title'] == channel]
channel_stats(df_last, 'Logan Paul Vlogs', num=10, arrange_by='likes')
def find_videos_by_trending_date(df, date, num=10, arrange_by='views', category=False):

    

    target_df = df.loc[df['trending_date'] == date][:num].sort_values(arrange_by, ascending=False)

    

    if category==True:

        cat_target = df.loc[df['trending_date'] == date].sort_values(arrange_by, ascending=False)

        cat = cat_target.groupby(['category'])['video_id'].count().sort_values(ascending=False).head()

        print('The categories with the most videos on this trending date:', cat)

    

    ax1 = target_df[['views']].plot.bar()

    

    ax2 = target_df[['likes', 'dislikes', 'comment_count']].plot.bar()

    

    labels = []

    for item in target_df['title']:

        labels.append(item[:10] + '...')

        

    ax1.set_title('Top {} views for videos trending on date {} arranged by {}'.format(num, date, arrange_by))

    ax1.set_xticklabels(labels, rotation=45)

    

    ax2.set_title('Top {} likes/dislikes/comments for videos trending on date {} arranged by {}'.format(num, date, arrange_by))

    ax2.set_xticklabels(labels, rotation=45)

    

    return target_df
find_videos_by_trending_date(df_last, '2017-11-14', 5, category=True)
def find_videos_by_publish_date(df, date, num=5, arrange_by='views', publish_to_trend_time=False):

    

    target_df = df.loc[df['publish_date'] == date][:num].sort_values(arrange_by, ascending=False)

    

    if publish_to_trend_time==True:

        target_df.insert(6, 'publish_to_trend_time', target_df['trending_date'] - target_df['publish_date'])

    

    ax1 = target_df[['views']].plot.bar()

    

    ax2 = target_df[['likes', 'dislikes', 'comment_count']].plot.bar()

    

    labels = []

    for item in target_df['title']:

        labels.append(item[:10] + '...')

        

    ax1.set_title('Top {} views for videos published on date {} arranged by {}'.format(num, date, arrange_by))

    ax1.set_xticklabels(labels, rotation=45)

    

    ax2.set_title('Top {} likes/dislikes/comments for videos published on date {} arranged by {}'.format(num, date, arrange_by))

    ax2.set_xticklabels(labels, rotation=45

                       )

    return target_df
find_videos_by_publish_date(df_last, '2017-11-13', publish_to_trend_time=True)
find_videos_by_publish_date(df_last, '2017-11-10', 2, 'comment_count')
def find_videos_by_category(df, cat, num=5, arrange_by='views'):

    

    target_df = df.loc[df['category'] == cat][:num].sort_values(arrange_by, ascending=False)

    

    ax1 = target_df[['views']].plot.bar()

    

    ax2 = target_df[['likes', 'dislikes', 'comment_count']].plot.bar()

    

    labels = []

    for item in target_df['title']:

        labels.append(item[:10] + '...')

        

    ax1.set_title('Top {} views for videos in category {} arranged by {}'.format(num, cat, arrange_by))

    ax1.set_xticklabels(labels, rotation=45)

    

    ax2.set_title('Top {} likes/dislikes/comments for videos in category {} arranged by {}'.format(num, cat, arrange_by))

    ax2.set_xticklabels(labels, rotation=45)

    

    return target_df
find_videos_by_category(df_last, 'Entertainment', 5)