# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline
import cufflinks as cf
cf.go_offline()
from __future__ import division
us_videos = pd.read_csv('../input/USvideos.csv')
us_videos.head(2)
us_cat = pd.read_json('../input/US_category_id.json')
us_cat.head(2)
us_videos.info()
us_cat.info()
us_cat['etag']
us_videos.head(2)
us_videos.shape
# transforming tending_date to date time col
us_videos['trending_date']=pd.to_datetime(us_videos['trending_date'],format='%y.%d.%m').dt.date
#splitting pulish time col into two seperate col i.e. time and hour
published_time = pd.to_datetime(us_videos['publish_time'],format='%Y-%m-%dT%H:%M:%S.%fZ')
us_videos['publish_time'] = published_time.dt.time
us_videos['publishDate'] = published_time.dt.date
us_videos['publishHour'] = published_time.dt.hour
us_videos.drop(['publish_time'],axis=1,inplace=True)
us_videos.head(10)
us_cat['items'][0]
categories = {category['id']: category['snippet']['title'] for category in us_cat['items']}
us_videos.insert(4,'category',us_videos['category_id'].astype(str).map(categories))


us_videos.head(3)
us_videos['dislike_percentage'] = us_videos['dislikes']/(us_videos['dislikes']+us_videos['likes'])
us_videos_first = us_videos.drop_duplicates(subset=['video_id'], keep='first', inplace=False)
us_videos_last = us_videos.drop_duplicates(subset=['video_id'], keep='last', inplace=False)

print("us_videos dataset contains {} videos".format(us_videos.shape[0]))
print("us_videos_first dataset contains {} videos".format(us_videos_first.shape[0]))
print("us_videos_last dataset contains {} videos".format(us_videos_last.shape[0]))
us_videos.shape
us_videos_first['time_to_trend'] = (us_videos_first.trending_date-us_videos_first.publishDate)/np.timedelta64(1,'D')
us_videos_first.head()
us_videos_first.plot(x='title',y='time_to_trend',kind='bar')
# which video trending the most
from IPython.display import HTML,display
select_cols = ['title','channel_title','thumbnail_link','category','publishDate']
# group video_id with above selected cols
#Using the .agg function allows you to calculate the frequency for each group using the standard library function len.
most_popular = us_videos.groupby(select_cols)['video_id'].agg({'code_count': len}).sort_values('code_count',ascending=False,).head(10).reset_index()
# Construction of HTML table with thumbnail images assigned to the most popular videos
# print(most_popular)

table_content =''
max_content_length = 50
for date, row in most_popular.T.iteritems():
    HTML_row = '<tr>'
    HTML_row += '<td><img src="' + str(row[2]) + '"style="width:100px;height:100px;"></td>'
    HTML_row += '<td>' + str(row[0]) + '</td>'
    HTML_row += '<td>' + str(row[1])  + '</td>'
    HTML_row += '<td>' + str(row[3]) + '</td>'
    HTML_row += '<td>' + str(row[4]) + '</td>'
    table_content += HTML_row + '</tr>'

display(HTML(
    '<table><tr><th>Photo</th><th>Channel Name</th><th style="width:250px;">Title</th><th>Category</th><th>Publish Date</th></tr>{}</table>'.format(table_content))
)

# initialize a list whish will store counters for subsequent publish hour

publish_h = [0]*24

for index,row in us_videos_first.iterrows():
    publish_h[row['publishHour']] +=1
    
values = publish_h
ind = np.arange(len(values))   #Here the length of your df is being used to set the stop param:
fig = sns.barplot(ind,values)
max_title_length = 30
number_of_creators = 25

top_creators = us_videos.groupby(['channel_title'])['channel_title'].agg(
    {"code_count": len}).sort_values(
    "code_count", ascending=False
).head(number_of_creators).reset_index()
channel_title = us_videos_first['channel_title']
print(top_creators)
