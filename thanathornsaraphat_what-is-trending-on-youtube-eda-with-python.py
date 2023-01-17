import pandas as pd
import numpy as np

import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

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
keep_columns = ['views', 'likes', 'dislikes', 'comment_count'] # only looking at correlations between these variables
corr_matrix = my_df[keep_columns].corr()
corr_matrix
fig, ax = plt.subplots()
heatmap = ax.imshow(corr_matrix, interpolation='nearest', cmap=cm.coolwarm)

# making the colorbar on the side
cbar_min = corr_matrix.min().min()
cbar_max = corr_matrix.max().max()
cbar = fig.colorbar(heatmap, ticks=[cbar_min, cbar_max])

# making the labels
labels = ['']
for column in keep_columns:
    labels.append(column)
    labels.append('')
ax.set_yticklabels(labels, minor=False)
ax.set_xticklabels(labels, minor=False)

plt.show()
print(my_df.shape)
my_df = my_df[~my_df.index.duplicated(keep='last')]
print(my_df.shape)
my_df.index.duplicated().any()
def visualize_most(my_df, column, num=10): # getting the top 10 videos by default
    sorted_df = my_df.sort_values(column, ascending=False).iloc[:num]
    
    ax = sorted_df[column].plot.bar()
    
    # customizes the video titles, for asthetic purposes for the bar chart
    labels = []
    for item in sorted_df['title']:
        labels.append(item[:10] + '...')
    ax.set_xticklabels(labels, rotation=45, fontsize=10)
    
    plt.show()
visualize_most(my_df, 'views')
visualize_most(my_df, 'likes', num=5) # only visualizes the top 5
visualize_most(my_df, 'dislikes')
visualize_most(my_df, 'comment_count')
def visualize_statistics(my_df, id_list): # taking a list of video ids
    target_df = my_df.loc[id_list]
    
    ax = target_df[['views', 'likes', 'dislikes', 'comment_count']].plot.bar()
    
    # customizes the video titles, for asthetic purposes for the bar chart
    labels = []
    for item in target_df['title']:
        labels.append(item[:10] + '...')
    ax.set_xticklabels(labels, rotation=45, fontsize=10)
    
    plt.show()

def visualize_like_dislike(my_df, id_list):
    target_df = my_df.loc[id_list]
    
    ax = target_df[['likes', 'dislikes']].plot.bar(stacked=True)
    
    # customizes the video titles, for asthetic purposes for the bar chart
    labels = []
    for item in target_df['title']:
        labels.append(item[:10] + '...')
    ax.set_xticklabels(labels, rotation=45, fontsize=10)
    
    plt.show()
sample_id_list = my_df.sample(n=10, random_state=4).index # creates a random sample of 10 video IDs
sample_id_list
sample_id_list = my_df.sample (n = 10, random_state = 4) .index # สร้างตัวอย่างสุ่มของรหัสวิดีโอ 10 รายการ
sample_id_list
visualize_statistics(my_df, sample_id_list)
visualize_like_dislike(my_df, sample_id_list)
my_df['dislikes'].plot.hist()

plt.show()
my_df['comment_count'].plot.hist()

plt.show()
my_df[['likes', 'dislikes']].plot.hist(alpha=0.5)

plt.show()
my_df[['dislikes', 'comment_count']].plot.hist(alpha=0.5)

plt.show()
category_count = my_df['category'].value_counts() # frequency for each category
category_count
ax = category_count.plot.bar()
ax.set_xticklabels(labels=category_count.index, rotation=45, fontsize=10)

plt.show()
