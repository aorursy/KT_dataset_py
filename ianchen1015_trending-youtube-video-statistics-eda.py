# Trending YouTube Video Statistics EDA
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd 
pd.options.display.max_rows = 10
from IPython.display import display, HTML

# remove warnings
import warnings
warnings.filterwarnings('ignore')
# read data
data= pd.read_csv("../input/USvideos.csv")
print (data.shape)
display(HTML(data.head().to_html()))
display(HTML(data.describe().to_html()))
# read category_id
import json
data_id = {}
with open('../input/US_category_id.json', 'r') as f:
    category_id = json.load(f)
    for i in category_id['items']:
        data_id[i['id']] = i['snippet']['title']
print (data_id)
# add 'category' column into data frame
category = []
for i in data['category_id']:
    category.append(data_id[str(i)])
data['category'] = pd.Series(category)
display(HTML(data.head(1).to_html()))
# correlation
corr_lst = ['views', 'likes', 'dislikes', 'comment_count']
corr_matrix = data[corr_lst].corr()
display(HTML(corr_matrix.to_html()))
plt.imshow(corr_matrix, interpolation='nearest')
# heatmap
plt.xticks((0, 1, 2, 3), ('views', 'likes', 'dislikes', 'comments'))
plt.yticks((0, 1, 2, 3), ('views', 'likes', 'dislikes', 'comments'))
plt.grid(True)
plt.colorbar()
plt.show()
# comment_count vs likes
plt.scatter(data['comment_count'].values, data['likes'].values, color='b')
plt.title('comment_count vs likes')
plt.xlabel('comment_count')
plt.ylabel('likes')
plt.show()

# comment_count vs dislikes
plt.scatter(data['comment_count'].values, data['dislikes'].values, color='b')
plt.title('comment_count vs dislikes')
plt.xlabel('comment_count')
plt.ylabel('dislikes')
plt.show()
# count the amount of each feature
def amount_rank(feature):
    ds = data.groupby(feature).size().sort_values(axis=0, ascending=False).head(50)
    ds.plot(kind='bar',stacked=False, figsize=(15,6))
    plt.title(feature)
    plt.ylabel('amount')
amount_rank('category')
amount_rank('channel_title')
# count the likes of features
def likes_rank(feature):
    df = data[[feature, 'likes']]
    df = df.sort_values(by='likes', ascending=False).head(30)
    df.plot(kind='bar',x=feature,y='likes', stacked=False, figsize=(15,6))
    plt.title(feature)
    plt.ylabel('likes')
likes_rank('channel_title')
likes_rank('title')
# likes rate of 30 most liked
def likes_rate_rank(feature):
    df = data[[feature, 'likes', 'dislikes']]
    df['likes_rate'] = df['likes']/(df['likes'] + df['dislikes'])
    df = df.sort_values(by='likes', ascending=False).head(30)
    df = df.sort_values(by='likes_rate', ascending=False)
    df.plot(kind='bar',x=feature,y='likes_rate', stacked=False, figsize=(15,6))
    plt.title(feature)
    plt.ylabel('likes')
likes_rate_rank('channel_title')
likes_rate_rank('title')
