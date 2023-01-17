import numpy as np 

import pandas as pd

import json



import seaborn as sns

sns.set_style('whitegrid')

import matplotlib.pyplot as plt



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



import nltk

from nltk.corpus import stopwords



from wordcloud import WordCloud, STOPWORDS



import warnings

warnings.filterwarnings(action="ignore")



pd.set_option('display.max_columns', 50)
df = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')
df.head()
df.info()
df.describe()
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df['publish_time'] = pd.to_datetime(df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
df.insert(5, 'publish_date', df['publish_time'].dt.date)
df['publish_time'] = df['publish_time'].dt.time
df['publish_date'] = pd.to_datetime(df['publish_date'])
id_to_cat = {}



with open('/kaggle/input/youtube-new/US_category_id.json', 'r') as f:

    data = json.load(f)

    for category in data['items']:

        id_to_cat[category['id']] = category['snippet']['title']
id_to_cat
df['category_id'] = df['category_id'].astype(str)
df.insert(5, 'category', df['category_id'].map(id_to_cat))
df.insert(8, 'publish_to_trend_days', df['trending_date'] - df['publish_date'])
df.insert(7, 'publish_month', df['publish_date'].dt.strftime('%m'))
df.insert(8, 'publish_day', df['publish_date'].dt.strftime('%a'))
df.insert(10, 'publish_hour', df['publish_time'].apply(lambda x: x.hour))
# Let's take a look at our dataframe with the new features

df.head()
df['video_id'].nunique()
len(df['video_id'])
print(df.shape)

df_last = df.drop_duplicates(subset=['video_id'], keep='last', inplace=False)

df_first = df.drop_duplicates(subset=['video_id'], keep='first', inplace=False)

print(df_last.shape)

print(df_first.shape)
print(df['video_id'].duplicated().any())

print(df_last['video_id'].duplicated().any())

print(df_first['video_id'].duplicated().any())
trend_days = df.groupby(['video_id'])['video_id'].agg(total_trend_days=len).reset_index()

df_last = pd.merge(df_last, trend_days, on='video_id')
df_last.head()
df_last.info()
def top_10(df, col, num=10):

    sort_df = df.sort_values(col, ascending=False).iloc[:num]

    

    fig = px.bar(sort_df, x=sort_df['title'], y=sort_df[col])

    

    labels = []

    for item in sort_df['title']:

        labels.append(item[:10] + '...')

        

    fig.update_layout(title = {'text':'Top {} videos with the highest {}'.format(num, col),

                           'y':0.95,

                           'x':0.4,

                            'xanchor':'center',

                            'yanchor':'top'},

                 xaxis_title='',

                 yaxis_title=col,

                     xaxis = dict(ticktext=labels))

  

    fig.show()

    

    return sort_df[['video_id', 'title', 'channel_title','category', col]]
top_10(df_last, 'views', 10)
top_10(df_last, 'likes')
top_10(df_last, 'dislikes')
top_10(df_last, 'comment_count')
def bottom_10(df, col, num=10):

    

    if col == 'likes' or col == 'dislikes':

        sort_df = df[df['ratings_disabled'] == False].sort_values(col, ascending=True).iloc[:num]

    elif col == 'comment_count':

        sort_df = df[df['comments_disabled'] == False].sort_values(col, ascending=True).iloc[:num]

    else:

        sort_df = df.sort_values(col, ascending=True).iloc[:num]

    

    fig = px.bar(sort_df, x=sort_df['title'], y=sort_df[col])

    

    labels = []

    for item in sort_df['title']:

        labels.append(item[:10] + '...')

        

    fig.update_layout(title = {'text':'Bottom {} videos with the lowest {}'.format(num, col),

                           'y':0.95,

                           'x':0.4,

                            'xanchor':'center',

                            'yanchor':'top'},

                 xaxis_title='',

                 yaxis_title=col,

                     )

  

    fig.show()

    

    return sort_df[['video_id', 'title', 'channel_title','category', 'total_trend_days', 'publish_to_trend_days', 'views', 'likes', 'dislikes', 'comment_count', 'ratings_disabled', 'comments_disabled']]
bottom_10(df_last, 'views')
bottom_10(df_last, 'likes')
bottom_10(df_last, 'dislikes')
bottom_10(df_last, 'comment_count')
top_channels = df_last.groupby(['channel_title'])['channel_title'].agg(code_count=len).sort_values("code_count", ascending=False)[:20].reset_index()



fig = px.bar(top_channels, x=top_channels['channel_title'], y=top_channels['code_count'])



fig.update_layout(title = {'text':'Channels with the most trending videos',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='',

                 yaxis_title='Number of trending videos')



fig.show()
top_channels_views = df_last.groupby(['channel_title'])['views'].agg(total_views=sum).sort_values("total_views", ascending=False)[:20].reset_index()



fig = px.bar(top_channels_views, x=top_channels_views['channel_title'], y=top_channels_views['total_views'])



fig.update_layout(title = {'text':'Channels with the most views',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='',

                 yaxis_title='Total views')



fig.show()
top_channels_likes = df_last.groupby(['channel_title'])['likes'].agg(total_likes=sum).sort_values("total_likes", ascending=False)[:20].reset_index()



fig = px.bar(top_channels_likes, x=top_channels_likes['channel_title'], y=top_channels_likes['total_likes'])



fig.update_layout(title = {'text':'Channels with the most likes',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='',

                 yaxis_title='Total likes')



fig.show()
top_channels_dislikes = df_last.groupby(['channel_title'])['dislikes'].agg(total_dislikes=sum).sort_values("total_dislikes", ascending=False)[:20].reset_index()



fig = px.bar(top_channels_dislikes, x=top_channels_dislikes['channel_title'], y=top_channels_dislikes['total_dislikes'])



fig.update_layout(title = {'text':'Channels with the most dislikes',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='',

                 yaxis_title='Total dislikes')



fig.show()
top_channels_comments = df_last.groupby(['channel_title'])['comment_count'].agg(total_comments=sum).sort_values("total_comments", ascending=False)[:20].reset_index()



fig = px.bar(top_channels_comments, x=top_channels_comments['channel_title'], y=top_channels_comments['total_comments'])



fig.update_layout(title = {'text':'Channels With Most Comments',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='',

                 yaxis_title='Total Comments')



fig.show()
print("Views quantiles")

print(df_last['views'].quantile([.01,.25,.5,.75,.99]))

print('---------------------------')

print('Likes quantiles')

print(df_last['likes'].quantile([.01,.25,.5,.75,.99]))

print('---------------------------')

print('Disikes quantiles')

print(df_last['dislikes'].quantile([.01,.25,.5,.75,.99]))

print('---------------------------')

print('Comments quantiles')

print(df_last['comment_count'].quantile([.01,.25,.5,.75,.99]))

print('---------------------------')
df_last['views_log'] = np.log(df_last['views'] + 1)

df_last['likes_log'] = np.log(df_last['likes'] + 1)

df_last['dislikes_log'] = np.log(df_last['dislikes'] + 1)

df_last['comments_log'] = np.log(df_last['comment_count'] + 1)
fig = px.histogram(df_last, x=df_last['category'])



fig.update_layout(title = {'text':'Number of videos sorted by category',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='',

                 yaxis_title='Count',

                 template='seaborn')



fig.show()
fig = px.box(df_last, x=df_last['category'], y=df_last['views_log'])



fig.update_layout(title = {'text':'Views distribution by category',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='',

                 yaxis_title='views_log',

                 template='seaborn')



fig.show()
fig = px.box(df_last, x=df_last['category'], y=df_last['likes_log'])



fig.update_layout(title = {'text':'Likes distribution by category',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='',

                 yaxis_title='likes_log',

                 template='seaborn')



fig.show()
fig = px.box(df_last, x=df_last['category'], y=df_last['dislikes_log'])



fig.update_layout(title = {'text':'Dislikes distribution by category',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='',

                 yaxis_title='dislikes_log',

                 template='seaborn')



fig.show()
fig = px.box(df_last, x=df_last['category'], y=df_last['comments_log'])



fig.update_layout(title = {'text':'Comments distribution by category',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='',

                 yaxis_title='comments_log',

                 template='seaborn')



fig.show()
video_trend = df_last.groupby('total_trend_days')['total_trend_days'].agg(count=len).sort_values('count', ascending=False).reset_index()



fig = px.bar(video_trend, x=video_trend['total_trend_days'], y=video_trend['count'])



fig.update_layout(title = {'text':'Number of videos arranged by trending days',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='Total trend days',

                 yaxis_title='Number of videos',

                 template='seaborn')



fig.show()   
com = 100.0*len(df_last[df_last['comments_disabled'] == True]) / len(df_last['comments_disabled'])

rat = 100.0*len(df_last[df_last['ratings_disabled'] == True]) / len(df_last['ratings_disabled'])

err = 100.0*len(df_last[df_last['video_error_or_removed'] == True]) / len(df_last['video_error_or_removed'])



fig = make_subplots(rows=1, cols=1)



fig.add_trace(go.Bar(x=[com], name='Percentage of videos with comments disabled'))

fig.add_trace(go.Bar(x=[rat], name='Percentage of videos with ratings disabled'))

fig.add_trace(go.Bar(x=[err], name='Percentage of videos with error or removed'))



fig.update_layout(title = {'text':'Percentage of videos with comments/ratings disabled or removed',

                           'y':0.9,

                           'x':0.4},

                 xaxis_title='Percentage',

                 yaxis_title='')



fig.show()
best_month = df_last.groupby('publish_month')['publish_month'].agg(count=len).sort_values('count', ascending=False).reset_index()



fig = px.bar(best_month, x=best_month['publish_month'], y=best_month['count'])



fig.update_layout(title = {'text':'Videos published arranged by months',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='Publish months',

                 yaxis_title='Count',

                 xaxis = dict(

        tickmode = 'array',

        tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

        ticktext = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))

                  



fig.show()
best_day = df_last.groupby('publish_day')['publish_day'].agg(count=len).sort_values('count', ascending=False).reset_index()



fig = px.bar(best_day, x=best_day['publish_day'], y=best_day['count'])



fig.update_layout(title = {'text':'Videos published arranged by day of the week',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='Publish days',

                 yaxis_title='Count') 



fig.show()
best_hour = df_last.groupby('publish_hour')['publish_hour'].agg(count=len).reset_index()



count = best_hour['count']

bins = [0, 155, 223, 327, 578]

labels = ['Bad','Decent','Good' ,'Great']



colors = {'Decent': 'orange',

          'Bad': 'red',

          'Good': 'lightgreen',

          'Great': 'darkgreen'}



# Build dataframe

color_df = pd.DataFrame({'y': count,

                   'x': range(len(count)),

                   'label': pd.cut(count, bins=bins, labels=labels)})



fig = go.Figure()



bars = []

for label, label_df in color_df.groupby('label'):

    bars.append(go.Bar(x=label_df.x,

                       y=label_df.y,

                       name=label,

                       marker={'color': colors[label]}))



go.FigureWidget(data=bars)

df_last['title_length'] = df_last['title'].apply(lambda x: len(x))
df_last['no_of_tags'] = df_last['tags'].apply(lambda x: len(x.split('|')))
fig = px.histogram(df_last, x=df_last['title_length'])



fig.update_layout(title = {'text':'Number of videos sorted by title length',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='Title length',

                 yaxis_title='Count')



fig.show()
fig = px.histogram(df_last, x=df_last['no_of_tags'])



fig.update_layout(title = {'text':'Number of videos sorted by number of tags',

                           'y':0.95,

                           'x':0.5},

                 xaxis_title='Number of Tags',

                 yaxis_title='Count')



fig.show()
eng_stopwords = set(stopwords.words('english'))
stopwords = set(STOPWORDS)



wordcloud = WordCloud(background_color='black',

                     stopwords=stopwords,

                     max_words=150,

                     max_font_size=40,

                     ).generate(str(df_last['title']))



plt.figure(figsize=(12,10))

plt.imshow(wordcloud)

plt.title('Most common words in title', fontsize=15)

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
stopwords = set(STOPWORDS)



wordcloud = WordCloud(background_color='white',

                     stopwords=stopwords,

                     #max_words=150,

                     max_font_size=50,

                     ).generate(str(df_last['tags']))



plt.figure(figsize=(12,10))

plt.imshow(wordcloud)

plt.title('Most common tags', fontsize=15)

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()