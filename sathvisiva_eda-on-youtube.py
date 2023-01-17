import pandas as pd
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from wordcloud import WordCloud, STOPWORDS
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected = True)
us_df = pd.read_csv('../input/USvideos.csv')
us_df.head()
us_df['trending_date'] = pd.to_datetime(us_df['trending_date'], format='%y.%d.%m')
us_df['publish_time'] = pd.to_datetime(us_df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
us_df.head()
us_df['publish_date'] = us_df['publish_time'].dt.date
us_df['publish_tym'] = us_df['publish_time'].dt.time
columns = ['views', 'likes' , 'dislikes' , 'comment_count']
for col in columns:
    us_df[col] = us_df[col].astype(int)
us_df['category_id'] = us_df['category_id'].astype(str)
us_df.info()
id_to_category = {}

with open('../input/US_category_id.json' , 'r') as f:
    data = json.load(f)
    for category in data['items']:
        id_to_category[category['id']] = category['snippet']['title']
id_to_category
us_df['category'] = us_df['category_id'].map(id_to_category)
us_df.head()
def view_bar(x,y,title):
    plt.figure(figsize = (13,11))
    sns.barplot(x = x, y = y)
    plt.title(title)
    plt.xticks(rotation = 90)
    plt.show()
x = us_df.category.value_counts().index
y = us_df.category.value_counts().values
title = "Categories"
view_bar(x,y,title)
x = us_df.channel_title.value_counts().head(10).index
y = us_df.channel_title.value_counts().head(10).values
title = "Top 10 Channels"
view_bar(x,y,title)
sort_by_views = us_df.sort_values(by ="views" , ascending = False).drop_duplicates('title', keep = 'first')
x = sort_by_views['title'].head(10)
y = sort_by_views['views'].head(10)
title = "Most watched videos"
view_bar(x,y,title)
sort_by_likes = us_df.sort_values(by ="likes" , ascending = False).drop_duplicates('title', keep = 'first')
x = sort_by_likes['title'].head(10)
y = sort_by_likes['likes'].head(10)
title = "Most liked videos"
view_bar(x,y,title)
sort_by_dislikes = us_df.sort_values(by ="dislikes" , ascending = False).drop_duplicates('title', keep = 'first')
x = sort_by_dislikes['title'].head(10)
y = sort_by_dislikes['dislikes'].head(10)
title = "Most disliked videos"
view_bar(x,y,title)
sort_by_comment = us_df.sort_values(by ="comment_count" , ascending = False).drop_duplicates('title', keep = 'first')
x = sort_by_comment['title'].head(10)
y = sort_by_comment['comment_count'].head(10)
title = "Most commented videos"
view_bar(x,y,title)
def createwordcloud(data , bgcolor , title):
    plt.figure(figsize = (13,14))
    wc = WordCloud(background_color = bgcolor, max_words = 1000, stopwords = STOPWORDS, max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')
tags = us_df['tags']
createwordcloud(tags , 'black' , 'commonly used tags' )
tags = us_df['tags'].map(lambda x : x.lower().split('|')).values
all_tags = [tag for t in tags for tag in t]
tags1 = pd.DataFrame({'tags' : all_tags})
x = tags1['tags'].value_counts().index[0:10]
y = tags1['tags'].value_counts().values[0:10]
title = "Top 10 most frequently used tags"
view_bar(x,y,title)

title = us_df['title']
createwordcloud(title , 'black' , 'commonly used words in titles' )
description = us_df['description'].astype('str')
createwordcloud(description , 'black' , 'commonly used words in description' )
us_df.head()
us_df['publish_date'] = pd.to_datetime(us_df['publish_date'])
us_df['diff'] = (us_df['trending_date'] - us_df['publish_date']).dt.days
us_df[['trending_date'  ,'views']].set_index('trending_date').plot()
sns.heatmap(us_df[['views' , 'likes' , 'dislikes' , 'comment_count']].corr(), annot = True, fmt = ".2f")
plt.show()
for i , category in us_df.groupby('category'):
    sns.heatmap(category[['views' , 'likes' , 'dislikes' , 'comment_count']].corr(), annot = True, fmt = ".2f")
    plt.title(i)
    plt.show()
dates_per_id = us_df[['video_id' , 'trending_date']].groupby('video_id', as_index = False).count()
long_trending = dates_per_id.loc[dates_per_id['trending_date'] == 13, 'video_id'].tolist()
long_trending_videos = us_df.loc[us_df['video_id'].isin(long_trending) , ['title' , 'trending_date' , 'views' , 'likes' , 'dislikes' , 'comment_count']]
long_trending_videos['views'] = long_trending_videos['views'].apply(lambda x : x / 100000)
video_titles = long_trending_videos['title'].unique().tolist()

views = []
likes = []
dislikes = []
comments = []
plots_list = [views, likes, dislikes, comments]
column_list = ['views' , 'likes' , 'dislikes' , 'comment_count']
boolen_list = [False , False, False, True]
color_list = []
for _ in range(0, len(video_titles)):
    color = 'rgb('+str(np.random.randint(1,256))+','+str(np.random.randint(1,256))+','+str(np.random.randint(1,256))+')'
    color_list.append(color)
    
for x in range(0 , len(plots_list)):
    for i in range(0, len(video_titles)):
        vt = video_titles[i]
        trace = go.Scatter(x = long_trending_videos.loc[long_trending_videos['title'] == vt , 'trending_date'],
                          y = long_trending_videos.loc[long_trending_videos['title'] == vt, column_list[x]],
                          name = vt,
                          line = dict(width = 2, color = color_list[i]),
                          legendgroup = vt,
                          showlegend = boolen_list[x])
        plots_list[x].extend([trace])
fig = tools.make_subplots(rows=4, cols=1, subplot_titles = ('Views', 'Comments', 'Likes', 'Dislikes'), vertical_spacing=0.07)
for i in views:
    fig.append_trace(i, 1, 1)
        
for i in comments:
    fig.append_trace(i, 2, 1)
        
for i in likes:
    fig.append_trace(i, 3, 1)
        
for i in dislikes:
    fig.append_trace(i, 4, 1)
    
fig['layout']['xaxis1'].update(title='')
fig['layout']['xaxis2'].update(title='')
fig['layout']['xaxis3'].update(title='')
fig['layout']['xaxis4'].update(title='')

fig['layout']['yaxis1'].update(title='mln. views')
fig['layout']['yaxis2'].update(title='comments')
fig['layout']['yaxis3'].update(title='likes')
fig['layout']['yaxis4'].update(title='dislikes')
    
fig['layout'].update(width=800, height=(1000 + len(video_titles)*60))
fig['layout'].update(title='Different metrics for videos')
fig['layout'].update(legend = dict(x=0.0,y = -(0.1+len(video_titles)*0.007),tracegroupgap = 1))

iplot(fig, filename='customizing-subplot-axes')
        
