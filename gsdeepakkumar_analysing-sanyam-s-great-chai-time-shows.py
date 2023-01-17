import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

from IPython.display import Image

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
kaggle=1

if kaggle==1:

    data=pd.read_csv('../input/chai-time-data-science/Episodes.csv')

    yt_thumbnail=pd.read_csv('../input/chai-time-data-science/YouTube Thumbnail Types.csv')

else:

    data=pd.read_csv('../data/Episodes.csv')

    yt_thumbnail=pd.read_csv('../data/YouTube Thumbnail Types.csv')
data.head()
data.shape
data[data['episode_id'].str.contains('M')][['episode_id','episode_name']]
yt_thumbnail.columns
yt_thumbnail[['youtube_thumbnail_type','description']]
data['youtube_thumbnail_type'].value_counts()
data=data.join(yt_thumbnail.set_index('youtube_thumbnail_type'),on='youtube_thumbnail_type',how='inner')
Image(filename = "../input/chai-time-data-science-thumbnails/youtube_default.PNG", width=300, height=300)
Image(filename = "../input/chai-time-data-science-thumbnails/youtube_default_ca.PNG", width=300, height=300)
Image(filename = "../input/chai-time-data-science-thumbnails/miniseries.PNG", width=300, height=300)
Image(filename = "../input/chai-time-data-science-thumbnails/custom_ctds.PNG", width=300, height=300)
plt.figure(figsize=(8,8))

ax=sns.boxplot(x=data['description'],y=data['youtube_impressions'],palette=sns.color_palette('Set2'))

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.title("Boxplot of impressions for each thumbnail type",fontsize=15)

plt.xlabel("Thumbnail",fontsize=12)

plt.ylabel("Youtube Impressions",fontsize=12)

plt.show()
plt.figure(figsize=(8,8))

ax=sns.boxplot(x=data['description'],y=data['youtube_views'],palette=sns.color_palette('Set2'))

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.title("Boxplot of total impression views for each thumbnail type",fontsize=15)

plt.xlabel("Thumbnail",fontsize=12)

plt.ylabel("Total Impression Views",fontsize=12)

plt.show()
plt.figure(figsize=(8,8))

sns.scatterplot(x=data['youtube_impressions'],y=data['youtube_impression_views'])

plt.title("Youtube Impressions Vs Youtube Impression Views",fontsize=15)

plt.xlabel("Impressions",fontsize=12)

plt.ylabel("Impression Views",fontsize=12)

plt.show()
data['category'].value_counts()
plt.figure(figsize=(8,8))

sns.barplot(x=data['category'],y=data['youtube_impressions'],palette=sns.color_palette('Set2'),ci=None)

plt.title("Which category has higher impression?",fontsize=15)

plt.xlabel("Category",fontsize=12)

plt.ylabel("Youtube Impressions",fontsize=12)
plt.figure(figsize=(8,8))

sns.barplot(x=data['category'],y=data['youtube_impression_views'],palette=sns.color_palette('Set2'),ci=None)

plt.title("Which category has higher impression views?",fontsize=15)

plt.xlabel("Category",fontsize=12)

plt.ylabel("Youtube Impression Views",fontsize=12)
plt.figure(figsize=(8,8))

sns.scatterplot(x=data['youtube_impressions'],y=data['youtube_impression_views'],hue=data['category'],palette='Set1')

plt.title("Youtube Impressions Vs Youtube Impression Views",fontsize=15)

plt.xlabel("Impressions",fontsize=12)

plt.ylabel("Impression Views",fontsize=12)

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(x=data['youtube_impressions'],y=data['youtube_ctr'])

plt.title("Youtube Impressions Vs Click Through Rate",fontsize=15)

plt.xlabel("Impressions",fontsize=12)

plt.ylabel("Click Through Rate",fontsize=12)

plt.show()
plt.figure(figsize=(8,8))

ax=sns.boxplot(x=data['description'],y=data['youtube_ctr'],palette=sns.color_palette('Set2'))

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.title("Boxplot of CTR for each thumbnail type",fontsize=15)

plt.xlabel("Thumbnail",fontsize=12)

plt.ylabel("CTR",fontsize=12)

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(x=data['youtube_impressions'],y=data['youtube_ctr'],hue=data['description'],palette='Set1')

plt.title("Youtube Impressions Vs Youtube CTR by thumbnail",fontsize=15)

plt.xlabel("Impressions",fontsize=12)

plt.ylabel("CTR",fontsize=12)

plt.show()
plt.figure(figsize=(8,8))

ax=sns.boxplot(x=data['category'],y=data['youtube_ctr'],palette=sns.color_palette('Set2'))

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.title("Boxplot of CTR for each category",fontsize=15)

plt.xlabel("Category",fontsize=12)

plt.ylabel("CTR",fontsize=12)

plt.show()
data['youtube_avg_duration_mins']=data['youtube_avg_watch_duration']/60
plt.figure(figsize=(8,8))

sns.distplot(data['youtube_avg_duration_mins'])

plt.title("Distribution of Average Watch Time",fontsize=15)

plt.xlabel("Average Watch Time",fontsize=12)

plt.show()
plt.figure(figsize=(8,8))

ax=sns.boxplot(x=data['category'],y=data['youtube_avg_duration_mins'],palette=sns.color_palette('Set2'))

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.title("Boxplot of Average watch duration (in mins) for each category",fontsize=15)

plt.xlabel("Category",fontsize=12)

plt.ylabel("Avg duration(mins)",fontsize=12)

plt.show()
data['episode_duration_mins']=data['episode_duration']/60
plt.figure(figsize=(8,8))

sns.distplot(data['episode_duration_mins'],bins=20)

plt.title("Distribution of Episode Duration",fontsize=15)

plt.xlabel("Episode Duration(in mins)",fontsize=12)

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(x=data['episode_duration_mins'],y=data['youtube_avg_duration_mins'])

plt.title("Episode Duration Vs Average Watchtime",fontsize=15)

plt.xlabel("Episode Duration(mins)",fontsize=12)

plt.ylabel("Average Watchtime(mins)",fontsize=12)

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(x=data['episode_duration_mins'],y=data['youtube_avg_duration_mins'],hue=data['category'],palette='Set1')

plt.title("Episode Duration Vs Average Watchtime by category",fontsize=15)

plt.xlabel("Episode Duration(mins)",fontsize=12)

plt.ylabel("Average Watchtime(mins)",fontsize=12)

plt.show()