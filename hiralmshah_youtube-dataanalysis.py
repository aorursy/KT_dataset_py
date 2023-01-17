# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
youtubeIN_df = pd.read_csv('/kaggle/input/youtube-new/INvideos.csv')

youtubeIN_df.head(3)
youtubeIN_df.set_index('trending_date', inplace = True)

youtubeIN_df[:3]
youtubeIN_df['views']
youtubeIN_df['views'].plot(figsize = (20,8))
youtubeIN_df[['likes','dislikes']].plot(figsize = (20,8))
youtubeIN_df.columns
youtubeIN_df['channel_title'].value_counts()
channel_title_count = youtubeIN_df['channel_title'].value_counts()[:30]
channel_title_count.plot(kind = 'bar', figsize = (15,8))
channelCategory = youtubeIN_df['category_id'].value_counts()

channelCategory
import json

with open('/kaggle/input/youtube-new/IN_category_id.json', 'r') as f:

    data = json.load(f)

youtubeIN_category = pd.DataFrame(data)



youtubeIN_category.head(5)
youtubeIN_category['items'][0]
id_to_category = {}



for c in youtubeIN_category['items']:

    id_to_category[int(c['id'])] = c['snippet']['title']

    

id_to_category
youtubeIN_df['category_title'] = youtubeIN_df['category_id'].map(id_to_category)

youtubeIN_df['category_title'][:3]
channel_category = youtubeIN_df['category_title'].value_counts()

channel_category
channel_category.plot(kind = 'bar', figsize = (20,8))
youtubeIN_df = youtubeIN_df.reset_index()

youtubeIN_df[:3]
youtubeIN_df = youtubeIN_df.reset_index()
youtubeIN_df['trending_date'] =  pd.to_datetime(youtubeIN_df['trending_date'], format='%y.%d.%m')
youtubeIN_df['trending_year'] = youtubeIN_df['trending_date'].dt.year

youtubeIN_df['trending_month'] = youtubeIN_df['trending_date'].dt.month

youtubeIN_df['trending_day'] = youtubeIN_df['trending_date'].dt.day

youtubeIN_df['is_weekend'] = youtubeIN_df['trending_day'].apply(lambda x:'weekend' if x > 4 else 'weekday') #1 means weekend
category_groupby = youtubeIN_df.groupby(['category_title','trending_year']).sum()

category_groupby                                   
category_groupby['views'].plot(kind = 'bar', figsize = (20,5))
category_groupby = youtubeIN_df.groupby(['category_title','is_weekend']).sum()
category_groupby['views']
category_groupby['views'].plot(kind = 'bar', figsize = (20,5))
entertainment_df = youtubeIN_df.loc[youtubeIN_df['category_title'] == 'Entertainment']

entertainment_df
entertainment_df.groupby('trending_year')['channel_title'].sum()
entertainment_df_2017 = entertainment_df.loc[entertainment_df['trending_year'] == 2017]

entertainment_df_2017
entertainment_df_2017.shape[0]
entertainment_df_2018 = entertainment_df.loc[entertainment_df['trending_year'] == 2018]

entertainment_df_2018
entertainment_df_2018.shape[0]
ent_popular_2017 = entertainment_df_2017.groupby('channel_title')[['views','likes','dislikes']].sum()[:30]



ent_popular_2017 = ent_popular_2017.sort_values('views', ascending=False)

ent_popular_2017.plot(kind = 'bar', figsize = (20,5))
ent_popular_2018 = entertainment_df_2018.groupby('channel_title')[['views','likes','dislikes']].sum()[:30]



ent_popular_2018 = ent_popular_2018.sort_values('views', ascending=False)

ent_popular_2018.plot(kind = 'bar', figsize = (20,5))
ent_popular_2017 = entertainment_df_2017.groupby('channel_title')[['views','likes','dislikes']].sum()

ent_channel_count_2017 = (entertainment_df_2017.groupby('channel_title').count())['index']

popular_entertainment_channels_2017 = pd.merge(ent_channel_count_2017, ent_popular_2017, how='right', on=['channel_title'])

popular_entertainment_channels_2017.rename(columns={"index": "video count"}, inplace = True)

popular_entertainment_channels_2017.sort_values('video count', ascending = False)[:50]
popular_entertainment_channels_2017.sort_values('video count', ascending = False)[:50].plot(kind = 'bar', figsize = (15,5))
ent_popular_2018 = entertainment_df_2018.groupby('channel_title')[['views','likes','dislikes']].sum()

ent_channel_count_2018 = (entertainment_df_2018.groupby('channel_title').count())['index']

popular_entertainment_channels_2018 = pd.merge(ent_channel_count_2018, ent_popular_2018, how='right', on=['channel_title'])

popular_entertainment_channels_2018.rename(columns={"index": "video count"}, inplace = True)

popular_entertainment_channels_2018.sort_values('video count', ascending = False)[:50]
popular_entertainment_channels_2018.sort_values('video count', ascending = False)[:50].plot(kind = 'bar', figsize = (15,5))