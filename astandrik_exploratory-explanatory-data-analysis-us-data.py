import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
df = pd.read_csv('../input/USvideos.csv', index_col=0, parse_dates=True, skipinitialspace=True)
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df.columns
id_dict = {}

with open('../input/US_category_id.json', 'r') as file:
    data = json.load(file)
    for category in data['items']:
        id_dict[int(category['id'])] = category['snippet']['title']

df['category_id'] = df['category_id'].map(id_dict)
df.head(3)
plt.figure(figsize=(12,6))
sns.countplot(x='category_id',data=df,palette='Blues_r', order=df['category_id'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Video Categories')
plt.ylabel('Total Videos by Categories')
plt.title('Total trending videos by Category in USA')

plt.tight_layout()
plt.show()
views_by_category_total = df.groupby('category_id')['views'].sum().sort_values(ascending=False)
views_by_category_avg = df.groupby('category_id')['views'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.barplot(y=views_by_category_total.index, x=(views_by_category_total/1000000), palette='Blues_r')
plt.xlabel('Total Views by Categories (millions)')
plt.ylabel('Video Categories')
plt.title('Total Views  of trending videos by Category in USA\n(in millions)')

plt.subplot(1,2,2)
sns.barplot(y=views_by_category_avg.index, x=(views_by_category_avg/1000000), palette='GnBu_d', alpha=0.6)
plt.xlabel('AVG Views by Categories (millions)')
plt.ylabel('')
plt.title('Average Views of trending videos by Category in USA\n(in millions)')

plt.tight_layout()
plt.show()
most_viewed_music = df[df['category_id']=='Music'][['title','category_id','views']].sort_values(by='views', ascending=False)
most_viewed_music = most_viewed_music.groupby('title')['views'].mean().sort_values(ascending=False).head(3)
most_viewed_music
unique_df_title = df.reset_index().groupby('title')['likes','dislikes'].mean()
unique_df_title['total_reviews'] = round(unique_df_title['likes'] + unique_df_title['dislikes'], 2)
unique_df_title = unique_df_title.sort_values(by='total_reviews', ascending=False).head(10)

unique_df_title[['likes','dislikes']].plot.bar(stacked=True, figsize=(12,6), color=['#55A868','#B66468'])
plt.xlabel('Video Title')
plt.ylabel('Frequency')
plt.title('Total Reviews (Likes/Dislikes) of Trending Videos')

plt.show()
plt.figure(figsize=(12,6))
sns.regplot(df['views']/1000000, df['likes']/1000, color='#55A868')
sns.regplot(df['views']/1000000, df['dislikes']/1000, color='#B66468')

plt.title('Views vs Likes/Dislikes of Trending Videos')
plt.xlabel('Views (in millions)')
plt.ylabel('Likes/Dislikes (in thousands)')
plt.legend(['Likes','Dislikes'])

plt.show()
top10_channels = df.groupby('channel_title')['title'].count().sort_values(ascending=False).head(20)

f = plt.figure(figsize=(12,6))
ax = f.add_subplot(111)
sns.barplot(y=top10_channels.index, x=top10_channels, palette='Reds_r', alpha=0.7)
plt.xlabel('Videos on Trending')
plt.ylabel('Channel Name')
plt.title('Channel with Most Trending Videos')
plt.xlim(0,150)

ax.text(0.70, 0.85, 'Top 5 channels with\ntrending videos',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='#3E4655', fontsize=15)
ax.text(0.68, 0.80, '3 of them are sports channel!',
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='#9A5B60', fontsize=13)

plt.tight_layout()
plt.show()
views_by_channel_total = df.groupby('channel_title')['views'].sum().sort_values(ascending=False).head(10)
views_by_channel_avg = df.groupby('channel_title')['views'].mean().sort_values(ascending=False).head(10)

f = plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.barplot(y=views_by_channel_total.index, x=(views_by_channel_total/1000000), palette='Blues_r')
plt.xlabel('Total Views by Channel (millions)')
plt.ylabel('Channel Name')
plt.title('Total Views of trending videos by Channel in USA\n(in millions)')

ax = plt.subplot(1,2,2)
sns.barplot(y=views_by_channel_avg.index, x=(views_by_channel_avg/1000000), palette='Reds_r', alpha=0.5)
plt.xlabel('AVG Views by Channel (millions)')
plt.ylabel('')
plt.title('Average Views of trending videos by Channel in USA\n(in millions)')

ax.text(0.35, 0.40, "5 of most avg views are\nfamous singer's channel",
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes,
        color='#3E4655', fontsize=15)

plt.tight_layout()
plt.show()
unique_df_channel = df.reset_index().groupby('channel_title')['likes','dislikes'].mean()
unique_df_channel['total_reviews'] = round(unique_df_channel['likes'] + unique_df_channel['dislikes'], 2)
unique_df_channel = unique_df_channel.sort_values(by='total_reviews', ascending=False).head(10)

unique_df_channel[['likes','dislikes']].plot.bar(stacked=True, figsize=(12,6), color=['#55A868','#B66468'])
plt.xlabel('Channel Title')
plt.ylabel('Frequency')
plt.title('Total Reviews (Likes/Dislikes) of Trending Videos in respective Channels')

plt.show()
spotlight = df[df['channel_title']=='YouTube Spotlight'].groupby('title')[['likes', 'dislikes']].mean()
spotlight.plot.barh(stacked=True, figsize=(12,6), color=['#55A868','#B66468'])
plt.xlabel('Number of Feedbacks (Likes/Dislikes)')
plt.ylabel('Video Title')
plt.title('YouTube Spotlight Videos Feedback')

plt.show()