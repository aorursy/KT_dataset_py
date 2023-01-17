import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

df
df.drop(["trending_date", "channel_title", "publish_time", "thumbnail_link", "comments_disabled", 
         "ratings_disabled", "video_error_or_removed", "description"], axis=1, inplace=True)
df.head()
df.index.size # Row count
len(df.index) # Row count (alternative)
df.columns.size # Column count
len(df.columns) # Column count (alternative)
df["likes"].mean()
df["dislikes"].mean()
df[(df["views"] == df["views"].max())] # Video with most views
df[(df["views"] == df["views"].max())]["title"] # Title of video with most views
df[(df["views"] == df["views"].max())]["title"].iloc[0] # String value of title of video with most views
df[(df["views"] == df["views"].min())] # Video with least views
df[(df["views"] == df["views"].min())]["title"] # Title of video with least views
df[(df["views"] == df["views"].min())]["title"].iloc[0] # String value of title of video with least views
df.groupby("category_id")["comment_count"].mean()
df["category_id"].value_counts()
df["title_length"] = df["title"].apply(len)

df
def count_tags(tag):
    return len(tag.split(sep="|"))
df["tag_count"] = df["tags"].apply(count_tags)

df
df.sort_values("likes", ascending=False)
df["likes"].iteritems() # iteritems creates a tuple that contains likes information for each row
list(df["likes"].iteritems()) # Lets see the structure
""" Function to calculcate likes / dislikes ratio for each row """
def get_likes_dislikes_ratio(likes, dislikes):
    likes_list = []
    for key, value in likes.iteritems():
        likes_list.append(value)
        
    dislikes_list = []
    for key, value in dislikes.iteritems():
        dislikes_list.append(value)
        
    ratio_list = []
    for likes_count, dislikes_count in list(zip(likes_list, dislikes_list)):
        if likes_count == 0 and dislikes_count == 0:
            ratio_list.append(0)
        else:
            ratio_list.append(likes_count / (likes_count + dislikes_count))
    
    return ratio_list
df["likes_dislikes_ratio"] = get_likes_dislikes_ratio(df["likes"], df["dislikes"])

df
df.sort_values("likes_dislikes_ratio", ascending=False) # sort by like / dislike ratio