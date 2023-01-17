import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import json

from datetime import datetime
youtube = pd.read_csv("../input/youtube-new/USvideos.csv")



youtube['trending_date'] = pd.to_datetime(youtube['trending_date'], format='%y.%d.%m') #parsing

youtube['publish_time'] = pd.to_datetime(youtube['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

youtube['category_id'] = youtube['category_id'].astype(str)



youtube.head()
youtube.info()
youtube.describe()
id_to_category = {}



with open('../input/youtube-new/US_category_id.json' , 'r') as f:

    data = json.load(f)

    for category in data['items']:

        id_to_category[category['id']] = category['snippet']['title']

        

youtube['category'] = youtube['category_id'].map(id_to_category)
youtube["ldratio"] = youtube["likes"] / youtube["dislikes"]
youtube["perc_comment"] = youtube["comment_count"] / youtube["views"]

youtube["perc_reaction"] = (youtube["likes"] + youtube["dislikes"]) / youtube["views"]
youtube['publish_date'] = youtube['publish_time'].dt.date

youtube['publish_tym'] = youtube['publish_time'].dt.time
youtube.head()
def distribution_cont(youtube, var):

    plt.hist(youtube[youtube["dislikes"] != 0][var])

    plt.xlabel(f"{var}")

    plt.ylabel("Count")

    plt.title(f"Distribution of Trending Video {var}")

    plt.show()

for i in ["views", "likes", "dislikes", "comment_count", "ldratio", "perc_reaction", "perc_comment"]:

    distribution_cont(youtube, i)
contvars = youtube[["views", "likes", "dislikes", "comment_count", "ldratio", "perc_comment", "perc_reaction"]]

corr = contvars.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
by_channel = youtube.groupby(["channel_title"]).size().sort_values(ascending = False).head(10)

sns.barplot(by_channel.values, by_channel.index.values, palette = "rocket")

plt.title("Top 10 Most Frequent Trending Youtube Channels")

plt.xlabel("Video Count")

plt.show()
by_cat = youtube.groupby(["category"]).size().sort_values(ascending = False)

sns.barplot(by_cat.values, by_cat.index.values, palette = "rocket")

plt.title("Most Frequent Trending Youtube Categories")

plt.xlabel("Video Count")

plt.show()
top_channels2 = youtube.groupby("channel_title").size().sort_values(ascending = False)

top_channels2 = list(top_channels2[top_channels2.values >= 20].index.values)

only_top2 = youtube

for i in list(youtube["channel_title"].unique()):

    if i not in top_channels2:

        only_top2 = only_top2[only_top2["channel_title"] != i]



by_views = only_top2.groupby(["channel_title"]).mean().sort_values(by = "views", ascending = False).head(10)

sns.barplot(by_views["views"], by_views.index.values, palette = "rocket")

plt.title("Top 10 Most Viewed Trending Youtube Channels")

plt.xlabel("Average Views")

plt.show()
by_views_cat = youtube.groupby(["category"]).mean().sort_values(by = "views", ascending = False)

sns.barplot(by_views_cat["views"], by_views_cat.index.values, palette = "rocket")

plt.title("Top 10 Most Viewed Trending Youtube Channels")

plt.xlabel("Average Views")

plt.show()
top_channels = youtube.groupby("channel_title").size().sort_values(ascending = False)

top_channels = list(top_channels[top_channels.values >= 20].index.values)

only_top = youtube

for i in list(youtube["channel_title"].unique()):

    if i not in top_channels:

        only_top = only_top[only_top["channel_title"] != i]



like_channel = only_top[only_top["dislikes"] != 0].groupby(["channel_title"]).mean().sort_values(by = "ldratio", ascending = False).head(10)

sns.barplot(like_channel["ldratio"], like_channel.index.values, palette = "rocket")

plt.title("Top 10 Most Liked Trending Youtube Channels")

plt.xlabel("Average Like to Dislike Ratio")

plt.show()
like_category = youtube[youtube["dislikes"] != 0].groupby("category").mean().sort_values(by = "ldratio", ascending = False)

sns.barplot(like_category["ldratio"], like_category.index.values, palette = "rocket")

plt.title("Top 10 Most Liked Trending Youtube Categories")

plt.xlabel("Average Like to Dislike Ratio")

plt.show()
def over_time(youtube, var):

    averages = youtube[youtube["dislikes"] != 0].groupby("trending_date").mean()

    plt.plot(averages.index.values, averages[var])

    plt.xticks(rotation = 90)

    plt.xlabel("Date")

    plt.ylabel(f"Average {var}")

    plt.title(f"Average {var} Over Time (11/14/17 - 6/14/18)")

    plt.show()
over_time(youtube, "views")
over_time(youtube, "ldratio")
over_time(youtube, "perc_reaction") #Recall perc_reaction is (likes + dislikes) / views
over_time(youtube, "perc_comment") #Recall perc_comment is comments / views
youtube["hour"] = youtube['publish_time'].dt.hour

by_hour = youtube.groupby("hour").mean()



plt.plot(by_hour.index.values, by_hour["views"])

plt.scatter(by_hour.index.values, by_hour["views"])

plt.xlabel("Hour of Day")

plt.ylabel("Average Number of Views")

plt.title("Average Amount of Views on Trending Videos by the Hour")

plt.show()
trump = youtube[youtube["title"].str.contains("Trump")]

trump.head()
trump.describe()
trump.sort_values(by = "ldratio").iloc[0]
trump.sort_values(by = "ldratio").iloc[2]
trump.sort_values(by = "ldratio").iloc[-1]
def get_top_video(youtube, min_trend_date, max_trend_date, stat, top = True, cat = list(youtube["category"].unique())):

    

    min_trend_list = min_trend_date.split("-")

    max_trend_list = max_trend_date.split("-")

    min_date = datetime(int(min_trend_list[0]), int(min_trend_list[1]), int(min_trend_list[2]))

    max_date = datetime(int(max_trend_list[0]), int(max_trend_list[1]), int(max_trend_list[2]))

    

    youtube = youtube[(youtube["trending_date"] >= min_date) & (youtube["trending_date"] <= max_date)]

    

    if stat == "ldratio":

        youtube = youtube[youtube["views"] >= 100000]

    

    for i in list(youtube["category"].unique()): 

        if i not in cat:

            youtube = youtube[youtube["category"] != i]

            

    if top == True:

        leaders = youtube.loc[youtube[stat].idxmax()][["title", "channel_title"]]

    else:

        leaders = youtube.loc[youtube[stat].idxmin()][["title", "channel_title"]]

    

    title_channel = list(leaders)

    

    return title_channel[0] + " by " + title_channel[1]
get_top_video(youtube, "2017-11-14", "2018-6-14", "views")
get_top_video(youtube, "2017-11-14", "2018-6-14", "likes")
get_top_video(youtube, "2017-11-14", "2018-6-14", "dislikes")
get_top_video(youtube, "2017-11-14", "2018-6-14", "ldratio")
get_top_video(youtube, "2017-11-14", "2018-6-14", "ldratio", top = False)
get_top_video(youtube, "2017-12-25", "2017-12-25", "views")
get_top_video(youtube, "2018-1-1", "2018-6-14", "views", cat = ["Gaming"])
get_top_video(youtube, "2018-1-1", "2018-6-14", "views", cat = ["Education", "Science & Technology"])
get_top_video(youtube, "2018-1-1", "2018-6-14", "perc_reaction", cat = ["Entertainment"])
get_top_video(youtube, "2018-1-1", "2018-6-14", "perc_comment", cat = ["News & Politics"])
get_top_video(youtube, "2018-1-1", "2018-6-14", "ldratio", top = False, cat = ["News & Politics"])