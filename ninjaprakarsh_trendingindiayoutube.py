import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
vids = pd.read_csv('../input/youtube-new/INvideos.csv')

categories = pd.read_json('../input/youtube-new/IN_category_id.json')
vids.head(3)
# Handling datatime data

vids["trending_date"] = pd.to_datetime(vids["trending_date"] , format="%y.%d.%m").dt.date

publish_time = pd.to_datetime(vids['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

vids['publish_date'] = publish_time.dt.date

vids['publish_time'] = publish_time.dt.time

vids['publish_hour'] = publish_time.dt.hour



vids.head()
categories.head()
categories["items"][0]
categories.shape , vids.shape
# Using dictionary compresension to map categories

cats ={int(cat["id"]):cat["snippet"]["title"] for cat in categories["items"]}



vids["categories"] = vids["category_id"].map(cats)
vids.head()
# Number of word in title



vids["title_length"] = vids["title"].apply(lambda x : len(x.replace(" " , "")))



# Number of external links in description

# We have heard youtube tends to favor videos with min external links, lets see if it effects trending

def ext_link_cnt(text):

    import re

    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

    return len(urls)



vids["ext_links"] = vids["description"].fillna(" ").apply(ext_link_cnt)
l_d_ratio = (vids["likes"]/vids["dislikes"]).dropna().median()

vids['dislike_percentage'] = (vids['dislikes']+1) / (vids['dislikes'] + vids['likes'] + l_d_ratio + 1)

vids['dislike_percentage'].head()
# Fixing skewness

vids['likes_log'] = np.log(vids['likes'] + 1)

vids['views_log'] = np.log(vids['views'] + 1)

vids['dislikes_log'] = np.log(vids['dislikes'] + 1)

vids['comment_log'] = np.log(vids['comment_count'] + 1)





# Views, Comments, Likes and Dislikes Visulization

plt.figure(figsize = (12,6))



plt.subplot(221)

g1 = sns.distplot(vids['views_log'])

g1.set_title("VIEWS LOG DISTRIBUITION", fontsize=16)



plt.subplot(224)

g2 = sns.distplot(vids['likes_log'],color='green')

g2.set_title('LIKES LOG DISTRIBUITION', fontsize=16)



plt.subplot(223)

g3 = sns.distplot(vids['dislikes_log'], color='r')

g3.set_title("DISLIKES LOG DISTRIBUITION", fontsize=16)



plt.subplot(222)

g4 = sns.distplot(vids['comment_log'])

g4.set_title("COMMENTS LOG DISTRIBUITION", fontsize=16)



plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.savefig("firstvisual.png" , bbox_inches="tight")

plt.show()
# Now several videos have been treading for many times, lets get rid of these for the purpose of our analysis

# to avoid duplicates



vids_last = vids.drop_duplicates(subset=['video_id'], keep='last', inplace=False)

vids_first = vids.drop_duplicates(subset=['video_id'], keep='first', inplace=False)

vids_first.head(3)
# Missing Values

vids.isnull().sum()
null_data = vids[vids["categories"].isnull()]

null_data.head()
vids["categories"].fillna("Activism & Random", inplace = True) 

vids[vids["category_id"]  == 29]

vids[vids["category_id"]  == 29].tail(2)
vids["days_before_trend"] = (vids.trending_date - vids.publish_date) / np.timedelta64(1, 'D')

vids["days_before_trend"] = vids["days_before_trend"].astype(int)
vids.loc[(vids['days_before_trend'] < 1), 'days_before_trend'] = 1

vids["views_per_day"] = vids["views"] / vids["days_before_trend"]

vids["views_per_day"] = vids["views_per_day"]



vids.head()
vids.drop(["title", "description"] , axis = 1 , inplace=True)

vids.to_csv("preprocessed_vids.csv" , index=False)
plt.style.use("ggplot")
plt.figure(figsize=(10,8))



sns.barplot(vids_first["publish_hour"].value_counts().index , vids_first["publish_hour"].value_counts().values)



plt.title("Trending v/s Time published" , fontsize=25 , fontweight="bold")

plt.xlabel("Time published in 24 hr format" , fontsize=18)

plt.ylabel("Amount of videos that made it to trending" , fontsize=18)



plt.savefig("trendingvtime.png" , bbox_inches="tight")

plt.show()
fig = plt.figure(figsize=(12,12))

ax = fig.add_subplot(111)



numerical_columns = [col for col in vids if vids[col].dtype in ["int64","float64"]]

sns.heatmap(vids[numerical_columns].corr("spearman"), annot=True, cmap="YlGnBu", ax=ax , cbar=False)



plt.savefig("featurecorr.png" , bbox_inches="tight")

plt.show()
fig = plt.figure(figsize=(10,8))



plt.plot(vids["title_length"].value_counts().sort_index().index, 

         vids["title_length"].value_counts().sort_index().values)



plt.title("Title Word Count v/s Trending" , fontsize=22 , fontweight="bold")

plt.xlabel("Title Word Count" , fontsize=18)

plt.ylabel("Number of videos treading", fontsize=18)



plt.savefig("titleword.png" , bbox_inches="tight")

plt.show()
fig = plt.figure(figsize=(10,8))



plt.plot(vids["ext_links"].value_counts().sort_index().index,

         vids["ext_links"].value_counts().sort_index().values)



plt.title("External Links in Video Discription v/s Trending" , fontsize=22 , fontweight="bold")

plt.xlabel("Number of External Links" , fontsize=18)

plt.ylabel("Number of Videos Treading", fontsize=18)



plt.savefig("ext_links.png" , bbox_inches="tight")

plt.show()
fig = plt.figure(figsize=(10,10))



ch_names = vids.groupby("channel_title")["video_id"].count().sort_values(ascending=False).index[:20]

cnts = vids.groupby("channel_title")["video_id"].count().sort_values(ascending=False).values[:20]



plt.barh(ch_names[::-1] , cnts[::-1])

plt.title("Most Trending Channels" , fontsize=24 , fontweight="bold")



plt.savefig("trending_channels.png" , bbox_inches="tight")

plt.show()
fig = plt.figure(figsize=(10,10))



topic_names = vids.groupby("categories")["video_id"].count().sort_values(ascending=False).index[:10]

cnts = vids.groupby("categories")["video_id"].count().sort_values(ascending=False).values[:10]



plt.barh(topic_names[::-1] , cnts[::-1])

plt.title("Most Trending Topics" , fontsize=24 , fontweight="bold")



plt.savefig("trending_topics.png" , bbox_inches="tight")

plt.show()
fig = plt.figure(figsize=(10,8))



plt.plot(vids["days_before_trend"].value_counts().sort_index().index,

        np.log(vids["days_before_trend"].value_counts().sort_index().values))



plt.xlabel("Days Befores Trending")

plt.ylabel("log(Number of Videos)" , fontstyle="italic")

plt.title("Days Taken to Trend" , fontsize=20 , fontweight="bold")



plt.savefig("days_taken.png" , bbox_inches="tight")

plt.show()
disliked_vids = vids.copy().sort_values(by="dislike_percentage" , ascending=False)[:20]

disliked_vids.sort_values(by="dislikes" , ascending=False , inplace=True)



fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)



sns.barplot(np.arange(0,20) , disliked_vids["dislikes"] , ax=ax , color="r")

sns.barplot(np.arange(0,20) , disliked_vids["likes"] , ax=ax , color="b")



plt.ylabel("Likes & Dislikes")

plt.xlabel("Videos")

plt.title("Videos with hightest dislike to like ratio")

plt.xticks([])



plt.savefig("hated.png" , bbox_inches="tight")

plt.show()
from wordcloud import WordCloud, STOPWORDS

from PIL import Image

import urllib

import requests





mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/O/i/x/Y/q/P/yellow-house-hi.png', stream=True).raw))



# This function takes in your text and your mask and generates a wordcloud. 

def generate_wordcloud(mask):

    word_cloud = WordCloud(width = 512, height = 512, background_color='white', stopwords=STOPWORDS, mask=mask).generate(str(vids["tags"]))

    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')

    plt.imshow(word_cloud)

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.savefig("wordcloud.png" , bbox_inches="tight")

    plt.show()

    

generate_wordcloud(mask)