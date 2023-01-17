import pandas as pd
import numpy as np

import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from datetime import datetime

matplotlib.rcParams['figure.figsize'] = (10, 10)
file_name = '../input/USvideos.csv' # change this if you want to read a different dataset
my_df = pd.read_csv(file_name, index_col='video_id')
my_df.head()
my_df['trending_date'] = pd.to_datetime(my_df['trending_date'], format='%y.%d.%m')
my_df['trending_date'].head()
my_df['publish_time'] = pd.to_datetime(my_df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
my_df['publish_time'].head()
# separates date and time into two columns from 'publish_time' column
my_df.insert(4, 'publish_date', my_df['publish_time'].dt.date)
my_df['publish_time'] = my_df['publish_time'].dt.time
my_df[['publish_date', 'publish_time']].head()
type_int_list = ['views', 'likes', 'dislikes', 'comment_count']
for column in type_int_list:
    my_df[column] = my_df[column].astype(int)

type_str_list = ['category_id']
for column in type_str_list:
    my_df[column] = my_df[column].astype(str)
# creates a dictionary that maps `category_id` to `category`
id_to_category = {}

with open('../input/US_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        id_to_category[category['id']] = category['snippet']['title']

id_to_category
my_df.insert(4, 'category', my_df['category_id'].map(id_to_category))
my_df[['category_id', 'category']].head()
mul_day_df = my_df[my_df.index.duplicated()]

print(mul_day_df.shape)
mul_day_df.head()
dup_index_set = list(set(mul_day_df.index))
len(dup_index_set)
freq_df = my_df.index.value_counts()
freq_df.head()
freq_df.plot.hist()

plt.show()
import matplotlib.patches as mpatches

def visualize_change(my_df, my_id):
    temp_df = my_df.loc[my_id]
    
    ax = plt.subplot(111)
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['views'], fmt='b-')
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['views'], fmt='bo')
    
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['likes'], fmt='g-')
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['likes'], fmt='go')
    
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['dislikes'], fmt='r-')
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['dislikes'], fmt='ro')
    
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['comment_count'], fmt='y-')
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['comment_count'], fmt='yo')
    
    patches = [
        mpatches.Patch(color='b', label='Views'),
        mpatches.Patch(color='g', label='Likes'),
        mpatches.Patch(color='r', label='Dislikes'),
        mpatches.Patch(color='y', label='Comments')
    ]
    
    plt.legend(handles=patches)
    
    plt.title(temp_df.iloc[0]['title'])
    
    plt.show()
# getting the id of the video that trended the longest
top_id = freq_df.index[0]
print(top_id)

visualize_change(my_df, top_id)
# getting a random video id
sample_id = freq_df.sample(n=1, random_state=4).index
print(sample_id)

visualize_change(my_df, sample_id)
# default values for single-entry videos
my_df['delta_views'] = my_df['views']
my_df['delta_likes'] = my_df['likes']
my_df['delta_dislikes'] = my_df['dislikes']
my_df['delta_comment_count'] = my_df['comment_count']
my_df['keep_trending'] = False
my_df.iloc[:5, -7:]
# has to have 2 rows or more
def get_delta_stat(video_id):
    temp_df = my_df.loc[video_id]
    
    temp_df.iloc[0, -1] = True

    for row_id in range(1, len(temp_df)):
        temp_df.iloc[row_id, -5] = temp_df.iloc[row_id]['views'] - temp_df.iloc[row_id - 1]['views'] # delta_views
        temp_df.iloc[row_id, -4] = temp_df.iloc[row_id]['likes'] - temp_df.iloc[row_id - 1]['likes'] # delta_likes
        temp_df.iloc[row_id, -3] = temp_df.iloc[row_id]['dislikes'] - temp_df.iloc[row_id - 1]['dislikes'] # delta_dislikes
        temp_df.iloc[row_id, -2] = temp_df.iloc[row_id]['comment_count'] - temp_df.iloc[row_id - 1]['comment_count'] # delta_comment_count
        temp_df.iloc[row_id, -1] = True # keep_trending

    temp_df.iloc[len(temp_df) - 1, -1] = False
    
    return temp_df
my_df.loc[sample_id]
sample_delta_df = get_delta_stat(sample_id)
sample_delta_df[['trending_date', 'views', 'likes', 'dislikes', 'comment_count', 'delta_views', 'delta_likes', 'delta_dislikes', 'delta_comment_count', 'keep_trending']]
my_df.loc[sample_id] = sample_delta_df
my_df.loc[sample_id][['trending_date', 'views', 'likes', 'dislikes', 'comment_count', 'delta_views', 'delta_likes', 'delta_dislikes', 'delta_comment_count', 'keep_trending']]
'''for video_id in freq_df[freq_df > 1].index:
    print(video_id)
    my_df.loc[video_id] = get_delta_stat(video_id)

my_df.head()[['trending_date', 'views', 'likes', 'dislikes', 'comment_count', 'delta_views', 'delta_likes', 'delta_dislikes', 'delta_comment_count', 'keep_trending']]'''
