import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from subprocess import check_output
from wordcloud import wordcloud, STOPWORDS
import warnings
from collections import Counter
import datetime
import glob
#hiding warnings for cleaner display
warnings.filterwarnings('ignore')

#Configuring some options
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
#For interactive plots
%matplotlib notebook
df = pd.read_csv("../input/youtube-new/USvideos.csv")
df_ca = pd.read_csv("../input/youtube-new/CAvideos.csv")
df_de = pd.read_csv("../input/youtube-new/DEvideos.csv")
df_fr = pd.read_csv("../input/youtube-new/FRvideos.csv")
df_gb = pd.read_csv("../input/youtube-new/GBvideos.csv")
PLOT_COLORS = ["#268bd2", "#0052CC", "#FF5722", "#b58900", "#003f5c"]
pd.options.display.float_format = '{:.2f}'.format
sns.set(style="ticks")
plt.rc('figure', figsize=(8, 5), dpi=100)
plt.rc('axes', labelpad=20, facecolor="#ffffff", linewidth=0.4, grid=True, labelsize=14)
plt.rc('patch', linewidth=0)
plt.rc('xtick.major', width=0.2)
plt.rc('ytick.major', width=0.2)
plt.rc('grid', color='#9E9E9E', linewidth=0.4)
plt.rc('font', family='Arial', weight='400', size=10)
plt.rc('text', color='#282828')
plt.rc('savefig', pad_inches=0.3, dpi=300)

df.head()
df.describe()
df.shape
df.info()
df[df["description"].apply(lambda x: pd.isna(x))].head(3)
cdf = df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts()\
.to_frame().reset_index()\
.rename(columns={"index": "year", "trending_date": "No_of_videos"})

fig, ax = plt.subplots()
_ = sns.barplot(x="year", y="No_of_videos", data = cdf, 
                palette = sns.color_palette(['#ff764a','#ffa600'], n_colors=7), ax=ax)
_ = ax.set(xlabel="Year", ylabel="No. of videos")
df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts(normalize=True)
columns_show = ['views', 'likes', 'dislikes', 'comment_count']
f, ax = plt.subplots(figsize=(8, 8))
dfe = df[columns_show].corr()
sns.heatmap(dfe,mask=np.zeros_like(dfe, dtype=np.bool), cmap='RdYlGn',linewidth=0.30,annot=True)
fig, ax = plt.subplots()
_ = plt.scatter(x=df['views'], y=df['likes'], color = PLOT_COLORS[2], edgecolors="#000000",
               linewidth=0.5)
_ = ax.set(xlabel="views", ylabel="likes")
fig, ax = plt.subplots()
_ = sns.distplot(df["views"], kde=False, color=PLOT_COLORS[4],
                hist_kws={'alpha': 1},bins=np.linspace(0, 2.3e8,47),ax=ax)
_ = ax.set(xlabel="Views",ylabel="No. of videos", xticks=np.arange(0, 2.4e8, 1e7))
_ = ax.set_xlim(right=2.5e8)
_ = plt.xticks(rotation=90)
fig, ax = plt.subplots()
_ = sns.distplot(df[df["views"]<25e6]["views"], kde=False, color=PLOT_COLORS[1], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Views", ylabel="No. of videos")
df[df['views']<1e6]['views'].count()/df['views'].count() * 100
plt.rc('figure.subplot',wspace=0.9)
fig, ax = plt.subplots()
_ = sns.distplot(df["likes"], kde=False, color=PLOT_COLORS[3],
                hist_kws={'alpha':1}, bins=np.linspace(0, 6e6, 61), ax=ax)
_ = ax.set(xlabel="Likes", ylabel="No. of Videos")
_ = plt.xticks(rotation=90)
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df['publish_time'] = pd.to_datetime(df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

#Separate date and time into two columns from 'publish_time' column
df.insert(4, 'pub_date', df['publish_time'].dt.date)
df['publish_time'] = df['publish_time'].dt.time
df['pub_date'] = pd.to_datetime(df['pub_date'])
us_views = df.groupby(['video_id'])['views'].agg('sum')
us_likes = df.groupby(['video_id'])['likes'].agg('sum')
us_dislikes = df.groupby(['video_id'])['dislikes'].agg('sum')
us_comment_count = df.groupby(['video_id'])['comment_count'].agg('sum')
df_usa_sdtr = df.drop_duplicates(subset='video_id', keep=False, inplace=False)
df_usa_mdtr = df.drop_duplicates(subset='video_id', keep='first', inplace=False)

frames = [df_usa_sdtr, df_usa_mdtr]
df_usa_without_duplicates = pd.concat(frames)

df_usa_comment_disabled = df_usa_without_duplicates[df_usa_without_duplicates['comments_disabled']==True].describe()
df_usa_rating_disabled = df_usa_without_duplicates[df_usa_without_duplicates['ratings_disabled']==True].describe()
df_usa_video_error = df_usa_without_duplicates[df_usa_without_duplicates['video_error_or_removed']==True].describe()
df_usa_sdtr.head()
df_usa_mdtr.head()
df_usa_mdtr = df.groupby(by=['video_id'],as_index=False).count().sort_values(by='title',ascending=False).head()

plt.figure(figsize=(8,8))
sns.set_style("whitegrid")
ax = sns.barplot(x=df_usa_mdtr['video_id'],y=df_usa_mdtr['trending_date'], data=df_usa_mdtr)
plt.xlabel("Video_Id")
plt.ylabel("Count")
plt.title("Top 5 videos that trended maximum days in USA")
df_us_max_views = us_views['j4KvrAUjn6c']
df_us_max_likes = us_likes['j4KvrAUjn6c']
df_us_max_dislikes = us_dislikes['j4KvrAUjn6c']
df_us_max_comment = us_comment_count['j4KvrAUjn6c']
cdf = df.groupby("channel_title").size().reset_index(name="video_count") \
.sort_values("video_count", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8,8))
_ = sns.barplot(x="video_count", y="channel_title", data=cdf, 
                palette=sns.cubehelix_palette(n_colors=20, reverse=True),ax=ax)
_ = ax.set(xlabel="No. of videos", ylabel="Channel")
usa_category_id = df_usa_without_duplicates.groupby(by=['category_id'],as_index=False).count().sort_values(by='title',ascending=False).head(5)

plt.figure(figsize=(7,7))
sns.kdeplot(usa_category_id['category_id']);
plt.xlabel("category IDs")
plt.ylabel("Count")
plt.title("Top 5 categories IDs for USA")
import simplejson as json
with open("../input/youtube-new/US_category_id.json") as f:
    categories = json.load(f)["items"]
cat_dict = {}
for cat in categories:
    cat_dict[int(cat["id"])] = cat["snippet"]["title"]
df['category_name'] = df['category_id'].map(cat_dict)
cdf = df["category_name"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "category_name", "category_name": "No_of_videos"}, inplace=True)
fig, ax = plt.subplots()
_ = sns.barplot(x="category_name", y="No_of_videos", data=cdf, 
                palette=sns.cubehelix_palette(n_colors=16, reverse=True), ax=ax)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
_ = ax.set(xlabel="Category", ylabel="No. of videos")
def Capitalized_word(s):
    for m in s.split():
        if m.isupper():
            return True
    return False

df["contains_capital_words"] = df["title"].apply(Capitalized_word)

value_counts = df["contains_capital_words"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No','Yes'],
          colors=['#003f5c', '#ffa600'], textprops={'color':'#040204'}, startangle=45)
_ = ax.axis('equal')
_ = ax.set_title('Title Contains Capitalized word?')
df["contains_capital_words"].value_counts(normalize=True)
df["title_length"] = df["title"].apply(lambda x: len(x))

fig, ax = plt.subplots()
_ = sns.distplot(df["title_length"], kde = False, rug = False, color=PLOT_COLORS[4], hist_kws={'alpha':1},
                ax=ax)
_ = ax.set(xlabel="Title Lenth", ylabel = "Number of Videos", xticks=range(0, 110, 10))
fig, ax = plt.subplots()
_ = ax.scatter(x=df['views'], y=df['title_length'], color=PLOT_COLORS[2], edgecolors="#000000", 
               linewidth=0.5)
_ = ax.set(xlabel = "views", ylabel = "Title_Length")
title_words = list(df["title"].apply(lambda x: x.split()))
title_words = [x for y in title_words for x in y]
Counter(title_words).most_common(25)
#wc = wordcloud.WordCloud(width=1200, height=600, collocations=False, Stopwords=None, 
#background_color="white",colormap="tab20b").generate_from_frequencies(dist(Counter(title_words).most_common(500)))

wc = wordcloud.WordCloud(width=1200, height=500, collocations=False, background_color="white",
                        colormap="tab20b").generate(" ".join(title_words))
plt.figure(figsize=(8,6))
plt.imshow(wc, interpolation='bilinear')
_ = plt.axis("off")
