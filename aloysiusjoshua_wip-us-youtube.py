# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import all important library

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

from scipy.stats import linregress

warnings.filterwarnings("ignore")
#read the file

USvideos_df = pd.read_csv("../input/youtube-new/USvideos.csv")

#create the dataframe for the category id

category_data = [[1,"Film & Animation"],[2,"Autos & Vehicles"],[10,"Music"],[15,"Pets & Animals"],

       [17,"Sports"],[18,"Short Movies"],[19,"Travel & Events"],[20,"Gaming"],

       [21,"Vlog"],[22,"People & Blogs"],[23,"Comedy"],[24,"Entertainment"],

       [25,"News & Politics"],[26,"Howto & Style"],[27,"Education"],

       [28,"Science & Technology"],[29,"Nonprofits & Activism"],[30,"Movies"],

       [31,"Anime/Animation"],[32,"Action/Adventure"],[33,"Classics"],

       [34,"Comedy"],[35,"Documentary"],[36,"Drama"],[37,"Family"],

       [38,"Foreign"],[39,"Horror"],[40,"Sci-Fo/Fantasy"],[41,"Thriller"],

       [42,"Shorts"],[43,"Shows"],[44,"Trailers"]]



category_df = pd.DataFrame(category_data, columns=["category_id","category_name"])

# print(category_df)
#describe the us videos

# print(USvideos_df.head())

# print(USvideos_df.describe()) 

print(USvideos_df.columns)
#total video available per category_name

cnt_video_per_category = USvideos_df.groupby(["category_id"]).count().reset_index()

cnt_video_per_category = cnt_video_per_category.loc[:,['category_id','video_id']]

df_1 = pd.merge(cnt_video_per_category,category_df,left_on='category_id',right_on='category_id',how='left')

df_1 = df_1.sort_values(by='video_id', ascending = False)

df_1["Proportion"] = round((df_1["video_id"]/sum(df_1["video_id"]) * 100),2)

print(df_1)



sns.set(style="whitegrid")

plt.figure(figsize=(11, 10))

ax = sns.barplot(x="category_name",y="video_id", data=df_1)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
#this section is used to see the trend of the youtube video published over time



#first we need to create new column containing date extracted from timestamp

x = USvideos_df['publish_time']

#convert the timestamp to datetime

df_datetime = pd.to_datetime(x)

#extract the date from the datetime

USvideos_df['date'] = df_datetime.dt.date

#extract the hour from the datetime

USvideos_df['hour'] = df_datetime.dt.hour

#extract the month from the datetime

USvideos_df['month'] = df_datetime.dt.month

#extract the year from the datetime

USvideos_df['year'] = df_datetime.dt.year

#============================================================================

#print(USvideos_df.head())



#create the dataframe

cnt_video_published_daily = USvideos_df.groupby(['year']).count().reset_index()

cnt_video_published_daily = cnt_video_published_daily.loc[:,['year','video_id']]

# print(cnt_video_published_daily)



sns.set(style="whitegrid")

plt.figure(figsize=(6,5))

sns.lineplot(x=cnt_video_published_daily["year"], y=cnt_video_published_daily["video_id"],data=cnt_video_published_daily, color="navy")

plt.show()
#this section is used to check the likes, dislike, and comment rate



#first we need to create 3 new variable

USvideos_df["likes_rate"] = USvideos_df["likes"] / USvideos_df["views"] * 100

USvideos_df["dislikes_rate"] = USvideos_df["dislikes"] / USvideos_df["views"] * 100

USvideos_df["comment_rate"] = USvideos_df["comment_count"] / USvideos_df["views"] * 100



#grouping the likes rate per category

cnt_likes_per_video_per_category = USvideos_df.groupby("category_id").mean().reset_index()

cnt_likes_per_video_per_category = cnt_likes_per_video_per_category.loc[:,['category_id','likes_rate','dislikes_rate','comment_rate']]



#left join to get the category name

df_2 = pd.merge(cnt_likes_per_video_per_category,category_df,left_on='category_id',right_on='category_id',how='left')

print(df_2)



#likes rate

df_2 = df_2.sort_values(by='likes_rate', ascending = False)

sns.set(style="whitegrid")

plt.figure(figsize=(11, 10))

plt.title("likes rate")

ax = sns.barplot(x="category_name",y="likes_rate", data=df_2)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()



#dislikes rate

df_2 = df_2.sort_values(by='dislikes_rate', ascending = False)

sns.set(style="whitegrid")

plt.figure(figsize=(11, 10))

plt.title("dislikes rate")

ax = sns.barplot(x="category_name",y="dislikes_rate", data=df_2)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()



#comments rate

df_2 = df_2.sort_values(by='comment_rate', ascending = False)

sns.set(style="whitegrid")

plt.figure(figsize=(11, 10))

plt.title("comments rate")

ax = sns.barplot(x="category_name",y="comment_rate", data=df_2)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
#this section is used to check the correlation between each metrics

plt.figure(figsize=(11, 8))



sns.heatmap(USvideos_df[['views', 'likes', 'dislikes', 'comment_count']].corr(), annot=True)

plt.show()