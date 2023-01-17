# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
yt_IN_df = pd.read_csv('/kaggle/input/youtube-new/INvideos.csv')
yt_IN_df.head(3)
yt_IN_df.set_index('trending_date', inplace = True)
yt_IN_df[:3]
yt_IN_df['views']
yt_IN_df['views'].plot(figsize = (20,7), color = 'blue')
yt_IN_df = yt_IN_df.reset_index()
yt_IN_df['trending_date'] = pd.to_datetime(yt_IN_df['trending_date'], format='%y.%d.%m')
yt_IN_df['trending_year'] = yt_IN_df['trending_date'].dt.year
yt_IN_df['trending_month'] = yt_IN_df['trending_date'].dt.month
yt_IN_df['trending_day'] = yt_IN_df['trending_date'].dt.day
yt_IN_df['trending_weekday'] = yt_IN_df['trending_date'].dt.weekday
yt_IN_df['is_weekend'] = yt_IN_df['trending_weekday'].apply(lambda x: 'weekend' if x>5 else 'weekday')
yr_weekend_grp = yt_IN_df.groupby(['trending_year','is_weekend']).mean()
yr_weekend_grp

yr_weekend_grp['views'].plot(figsize = (20,7), color = 'blue')
yt_IN_df['channel_title'].value_counts()[:20]

top_Chan = yt_IN_df['channel_title'].value_counts()[:20]
top_Chan.plot(kind = 'bar', figsize = (20,7), color = 'blue')
plt.title('20 Channels with Maximum posted videos')
yt_IN_df.reset_index()
chan_grp = yt_IN_df.groupby(['channel_title']).sum()
sort_chan_grp = chan_grp.sort_values(by = ['views'], ascending = False)['views']
sort_chan_grp[:10].plot(kind='bar', figsize=(20,7), color = 'blue')

sort_chan_grp[:10]
import json
with open('/kaggle/input/youtube-new/IN_category_id.json', 'r') as f:
    data = json.load(f)
yt_category = pd.DataFrame(data)

yt_category.head(5)
yt_category['items'][0]
id_to_cat = {}
for c in yt_category['items']:
    id_to_cat[int(c['id'])]=c['snippet']['title']
    
id_to_cat
yt_IN_df['Cat_title'] = yt_IN_df['category_id'].map(id_to_cat)
yt_IN_df[:5]
cat_grp = yt_IN_df.groupby(['Cat_title']).sum()
sort_cat_grp = cat_grp.sort_values(by = ['views'], ascending = False)['views']
sort_cat_grp
sort_cat_grp.plot(kind='bar', figsize = (20,7), color = 'blue')
plt.title('Views per Category title')
new_catgrp = yt_IN_df.groupby(['Cat_title']).agg({'category_id':'size', 'views':sum}).rename(columns={'category_id':'count_Videos','views':'total_Views'})
new_catgrp
new_catgrp['per vid view'] = new_catgrp['total_Views']/new_catgrp['count_Videos']
new_catgrp.sort_values(by = ['per vid view'], ascending = False)
game_yt_df = yt_IN_df[yt_IN_df['Cat_title'] == 'Gaming']
game_yt_df[:1]
view_game = game_yt_df.groupby(['channel_title']).agg({'channel_title':'count', 'views':sum}).rename({'channel_title': 'Count of Videos', 'views': 'Total Views'})
view_game['per vid view'] = view_game['views']/view_game['channel_title']
view_game.sort_values(by = 'per vid view', ascending = False)