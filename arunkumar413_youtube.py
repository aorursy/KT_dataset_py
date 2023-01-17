import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
us_videos = pd.read_csv("../input/USvideos.csv")
us_cat = pd.read_json("../input/US_category_id.json")

us_videos.head(2)
us_videos.dtypes
us_videos['trending_date']= pd.to_datetime(us_videos['trending_date'],format='%y.%d.%m')
us_videos['publish_time'] = pd.to_datetime(us_videos['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
us_videos.head()
us_videos.info()
us_videos.dtypes
us_videos['publish_year']= us_videos['publish_time'].dt.year
us_videos['publish_month']= us_videos['publish_time'].dt.month
us_videos['publish_day']= us_videos['publish_time'].dt.day
us_videos['publish_hour']= us_videos['publish_time'].dt.hour
us_videos.head(2)

metrics = ['likes','views','comment_count','dislikes']

grp_title = us_videos.groupby('title')[metrics].sum()
grp_title.reset_index(inplace=True)

grp_cat = us_videos.groupby('category_id')[metrics].sum()
grp_cat.reset_index(inplace=True)


likes = grp_title.sort_values('likes',ascending=True)
likes.reset_index(inplace=True)

views = grp_title.sort_values('views',ascending=True)
views.reset_index(inplace=True)


dislikes = grp_title.sort_values('dislikes',ascending=True)
dislikes.reset_index(inplace=True)


comment_count = grp_title.sort_values('comment_count',ascending=True)
comment_count.reset_index(inplace=True)


likes.tail(10).plot(kind='barh',x='title',y='likes',title= '10 most liked videos') 
views.tail(10).plot(kind='barh',x='title',y='views',title= '10 most viewed videos')
dislikes.tail(10).plot(kind='barh',x='title',y='dislikes',title= '10 most disliked videos')
comment_count.tail(10).plot(kind='barh',x='title',y='comment_count',title= '10 most commented videos')
fig = plt.figure(figsize=(30, 10))

ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4)

us_videos.likes.plot(kind='hist',bins=25,ax=ax1,title="likes")
us_videos.views.plot(kind='hist',bins=25,ax=ax2,title= 'views')
us_videos.dislikes.plot(kind='hist',bins=25,ax=ax3,title='dislikes')
us_videos.comment_count.plot(kind='hist',bins=25,ax=ax4,title='comment_count')
us_videos.likes.describe()
month_grp = us_videos.groupby('publish_month')[metrics].sum()
month_grp.reset_index(inplace=True)
fig = plt.figure(figsize=(30, 10))

ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4)


month_grp['likes'].plot(kind='bar',title='likes by hour',ax=ax1)
month_grp['dislikes'].plot(kind='bar',title='dislikes by hour',ax=ax2)
month_grp['views'].plot(kind='bar',title='views by hour',ax=ax3)
month_grp['comment_count'].plot(kind='bar',title='comments by hour',ax=ax4)
hour_grp = us_videos.groupby('publish_hour')[metrics].sum()
hour_grp.reset_index(inplace=True)
fig = plt.figure(figsize=(30, 10))

ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,2,3)
ax4 = plt.subplot(2,2,4)


hour_grp['likes'].plot(kind='bar',title='likes by hour',ax=ax1)
hour_grp['dislikes'].plot(kind='bar',title='dislikes by hour',ax=ax2)
hour_grp['views'].plot(kind='bar',title='views by hour',ax=ax3)
hour_grp['comment_count'].plot(kind='bar',title='comments by hour',ax=ax4)
