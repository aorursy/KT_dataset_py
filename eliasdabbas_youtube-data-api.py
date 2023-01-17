!pip install "advertools>=0.8.1"
import datetime



import advertools as adv

import pandas as pd

import plotly.graph_objects as go

import plotly





key = 'YOUR_GOOGLE_API_KEY'

pd.options.display.max_columns = None



print('package    version')

print('==================')

for package in [adv,pd, plotly]:

    print(f'{package.__name__:<12}', package.__version__)

print('\nLast updated:', format(datetime.datetime.now(), '%b %d, %Y'))
list(enumerate([func for func in dir(adv.youtube) if func[0] != '_'], 1))
# activities = adv.youtube.activities_list(key=key, part='snippet', channelId='UCBR8-60-B28hp2BmDPdntcQ') # see below how you can get an account's ID

activities = pd.read_csv('../input/youtube-data-api-datasets/activities.csv', parse_dates=['queryTime'])

print(activities.shape)

activities
activities[['snippet.channelTitle', 'snippet.publishedAt', 'snippet.type', 'snippet.title']]
activities.filter(regex='param_')
# yt_channels = adv.youtube.channels_list(key=key, part='snippet,statistics,contentDetails',

#                                         forUsername=['google', 'youtube', 'vevo'])

yt_channels = pd.read_csv('../input/youtube-data-api-datasets/yt_channels.csv', parse_dates=['queryTime'])

print(yt_channels.shape)

yt_channels
yt_channels.filter(regex='snippet\.title|statistics|query|published').set_index(['queryTime', 'snippet.publishedAt', 'snippet.title']).style.format('{:,}')
yt_channels.filter(regex='content')
# channels_activities = adv.youtube.activities_list(key=key, part='contentDetails,snippet',

#                                                   channelId=yt_channels['id'].tolist(), maxResults=200) 

channels_activities = pd.read_csv('../input/youtube-data-api-datasets/channels_activities.csv', parse_dates=['snippet.publishedAt'])

print(channels_activities.shape)

channels_activities.head(3)
channels_activities.groupby('snippet.channelTitle')['snippet.type'].value_counts().to_frame()
channels_activities['pub_month'] = [pd.Period(p, 'M') for p in channels_activities['snippet.publishedAt']]

channels_activities['pub_year'] = [pd.Period(p, 'A') for p in channels_activities['snippet.publishedAt']]

channels_activities.filter(regex='pub').head()
channels_activities['snippet.channelTitle'].unique()
fig = go.Figure()

for channel in ['YouTube', 'Google', 'Vevo']:

    df = (channels_activities

          [(channels_activities['snippet.channelTitle']==channel) & (channels_activities['snippet.type']=='upload')]

          .groupby(['pub_month', 'snippet.channelTitle'])

          ['snippet.type']

          .value_counts().to_frame()

          .rename(columns={'snippet.type': 'count'})

          .reset_index())

    fig.add_scatter(x=df['pub_month'].astype(str), y=df['count'], name=channel, mode='lines+markers', marker={'size': 10})



fig.layout.template = 'none'

fig.layout.title = 'Video Uploads per Month'

fig
fig = go.Figure()

for channel in ['YouTube', 'Google', 'Vevo']:

    df = (channels_activities

          [(channels_activities['snippet.channelTitle']==channel) & (channels_activities['snippet.type']=='upload')]

          .groupby(['pub_year', 'snippet.channelTitle'])

          ['snippet.type']

          .value_counts().to_frame()

          .rename(columns={'snippet.type': 'count'})

          .reset_index())

    fig.add_scatter(x=df['pub_year'].astype(str), y=df['count'], name=channel, marker={'size': 10})



fig.layout.template = 'none'

fig.layout.title = 'Video Uploads per Year'

fig
for channel in channels_activities['snippet.channelTitle'].unique():

    channel_df = channels_activities[channels_activities['snippet.channelTitle']==channel]

    val_counts = (channel_df['snippet.type'].value_counts()

                  .rename_axis(channel, axis=0))

    dates = channel_df['snippet.publishedAt']

    print(val_counts)

    print(' min date: ', dates.min().date(), ' | max date: ', dates.max().date(), ' | time delta: ', dates.max() - dates.min(), sep='')

    print('====\n')
# languages = adv.youtube.i18n_languages_list(key=key, part='snippet', hl=['en', 'fr', 'de'])

languages = pd.read_csv('../input/youtube-data-api-datasets/languages.csv', parse_dates=['queryTime'])

languages.head()
(languages

 .filter(regex='id|name|param_hl')

 .sort_values('id')[:12]

 .reset_index(drop=True))
# regions = adv.youtube.i18n_regions_list(key=key, part='snippet', hl=['en', 'ar', 'ja'])

regions = pd.read_csv('../input/youtube-data-api-datasets/regions.csv', parse_dates=['queryTime'])

regions.head()
(regions

 .filter(regex='id|name|param_hl')

 .sort_values('id')[:12]

 .reset_index(drop=True))
# channel_categories = adv.youtube.guide_categories_list(key=key, part='snippet', regionCode='us')

channel_categories = pd.read_csv('../input/youtube-data-api-datasets/channel_categories.csv', parse_dates=['queryTime'])

channel_categories.head(3)
channel_categories['snippet.title'].to_frame()
# video_categories = adv.youtube.video_categories_list(key=key, part='snippet', regionCode='us', hl=['en', 'es', 'zh-CN'])

video_categories = pd.read_csv('../input/youtube-data-api-datasets/video_categories.csv', parse_dates=['queryTime'])

video_categories.head(3)
(video_categories

 .iloc[pd.Series([(x,x+32,x+64) for x in range(32)]).explode()]

 [['id', 'snippet.title', 'param_hl']]

 [:15])
# comment_thrd_vid = adv.youtube.comment_threads_list(key=key, part='snippet', videoId='kJQP7kiw5Fk',

#                                                     maxResults=150,)

comment_thrd_vid = pd.read_csv('../input/youtube-data-api-datasets/comment_thrd_vid.csv', 

                               parse_dates=['queryTime', 'snippet.topLevelComment.snippet.publishedAt',

                                            'snippet.topLevelComment.snippet.updatedAt'])

comment_thrd_vid.head(3)
comment_thrd_vid['snippet.topLevelComment.snippet.publishedAt'].max() - comment_thrd_vid['snippet.topLevelComment.snippet.publishedAt'].min()
comment_thrd_vid.filter(regex='authorDisplay|textOriginal|likeCount|publishedAt')
(comment_thrd_vid

 .groupby('snippet.topLevelComment.snippet.authorDisplayName')

 .agg({'snippet.topLevelComment.snippet.likeCount': ['sum', 'count']})

 .sort_values(('snippet.topLevelComment.snippet.likeCount','sum'), ascending=False))
comment_thrd_vid['id'][:10]
# despacito = adv.youtube.videos_list(key=key, part='contentDetails,snippet,statistics', id='kJQP7kiw5Fk')



despacito = pd.read_csv('../input/youtube-data-api-datasets/despacito.csv', parse_dates=['queryTime', 'snippet.publishedAt'])

print(despacito.shape)

despacito
(despacito

 .filter(regex='statistics|queryTime')

 .set_index('queryTime')

 .style.format('{:,}'))
today = despacito['queryTime'][0]

print('Request date:', format(today, '%b %d, %Y'))

print("Despacito was published " + str((today - despacito['snippet.publishedAt'][0]).days) + " days before this request")
# vevo_playlists = adv.youtube.playlists_list(key=key, part='snippet,contentDetails', maxResults=200,

#                                             channelId='UC2pmfLm7iq6Ov1UwYrWYkZA')

vevo_playlists = pd.read_csv('../input/youtube-data-api-datasets/vevo_playlists.csv', parse_dates=['snippet.publishedAt', 'queryTime'])

print(vevo_playlists.shape)

vevo_playlists.head(2)
vevo_playlists['snippet.publishedAt'].max() - vevo_playlists['snippet.publishedAt'].min()
vevo_playlists.filter(regex='published|snippet\.title|description|itemCount|id$').head()
vevo_playlists['contentDetails.itemCount'].value_counts().to_frame().head()
fig = go.Figure()

fig.add_bar(x=vevo_playlists['contentDetails.itemCount'].value_counts().index,

            y=vevo_playlists['contentDetails.itemCount'].value_counts().values)



fig.layout.template = 'none'

fig.layout.xaxis.title = 'Number of videos per playlist'

fig.layout.yaxis.title = 'Count of playlists'

fig
(vevo_playlists

 .sort_values('contentDetails.itemCount', ascending=False)

 [['snippet.title', 'contentDetails.itemCount', 'id']])
# vevo_playlist_items = adv.youtube.playlist_items_list(key=key, part='snippet', maxResults=300,

#                                                       playlistId='PL9tY0BWXOZFsVNsIUCPoITIuZlzM3Y4bq')



vevo_playlist_items = pd.read_csv('../input/youtube-data-api-datasets/vevo_playlist_items.csv', parse_dates=['snippet.publishedAt', 'queryTime'])

print(vevo_playlist_items.shape)

vevo_playlist_items.head(2)
vevo_playlist_items.filter(regex='published|snippet\.(title|description)|videoId')
# vevo_subscriptions = adv.youtube.subscriptions_list(key=key, part='snippet,subscriberSnippet,contentDetails', 

#                                                     channelId='UC2pmfLm7iq6Ov1UwYrWYkZA', maxResults=250)



vevo_subscriptions = pd.read_csv('../input/youtube-data-api-datasets/vevo_subscriptions.csv', parse_dates=['snippet.publishedAt', 'queryTime'])

print(vevo_subscriptions.shape)

vevo_subscriptions.head(2)
vevo_subscriptions.filter(regex='published|title|description|Count')
# top10_world_vid = adv.youtube.videos_list(key=key,

#                                           part='snippet,statistics,contentDetails,topicDetails',

#                                           chart='mostPopular', 

#                                           regionCode=regions['id'].tolist(),

#                                           maxResults=10)

top10_world_vid = pd.read_csv('../input/youtube-data-api-datasets/top10_world_vid.csv', 

                              parse_dates=['snippet.publishedAt', 'queryTime'])

print(top10_world_vid.shape)

top10_world_vid.head(2)
category_dict = dict(video_categories.query('param_hl == "en"')[['id', 'snippet.title']].values)

top10_world_vid['category'] = [None if pd.isna(x) else category_dict[x] for x in top10_world_vid['snippet.categoryId']]

top10_world_vid[['snippet.categoryId', 'category']].head()
(top10_world_vid

 .sort_values(['statistics.viewCount', 'snippet.title', 'param_regionCode'], 

              ascending=False)

 [['snippet.title', 'statistics.viewCount', 'param_regionCode']]

 .head(15))
top10_world_vid['snippet.title'].value_counts().to_frame()[:10]
'errors' in top10_world_vid # a column named "errors" would be in the dataset if we had any
top10_world_vid['snippet.title'].isna().sum()
top10_world_vid[top10_world_vid['snippet.title'].isna()]
top10_world_vid[top10_world_vid['snippet.title'].isna()]['param_regionCode']
(top10_world_vid

 .query('param_regionCode=="BR"')

 .dropna(subset=['statistics.viewCount'])

 .sort_values('statistics.viewCount', ascending=False)

 [['snippet.title', 'statistics.viewCount']]

 .style.format({'statistics.viewCount': '{:,}'}))
(top10_world_vid

 .query('param_regionCode=="FR"')

 .dropna(subset=['statistics.viewCount'])

 .sort_values('statistics.viewCount', ascending=False)

 [['snippet.title', 'statistics.viewCount']]

 .style.format({'statistics.viewCount': '{:,}'}))
(top10_world_vid

 .dropna(subset=['statistics.viewCount'])

 .sort_values(['param_regionCode', 'statistics.viewCount'],

              ascending=[True, False])

 .groupby('param_regionCode', as_index=False)

 .head(3)

 [['param_regionCode', 'snippet.title',  'statistics.viewCount']][:15]

 .style.format({'statistics.viewCount': '{:,}'}))
category_sum_count = (top10_world_vid

                      .drop_duplicates(subset=['snippet.title'])

                      .dropna(subset=['statistics.viewCount'])

                      .groupby('category')

                      .agg({'statistics.viewCount': ['count', 'sum']})

                      ['statistics.viewCount']

                      .sort_values('sum', ascending=False))

category_sum_count.style.format({'sum': '{:,}'})
fig = go.Figure()

labels = (category_sum_count.index.astype(str) + ' (' + category_sum_count['count'].astype(str) + ' videos)').values

fig.add_treemap(labels=labels, 

                parents=['All Categories' for i in range(len(category_sum_count))], 

                values=category_sum_count['sum'],

                texttemplate='<b>%{label}</b><br><br>Total views: %{value}<br>%{percentParent} of total')

fig.layout.template = 'none'

fig.layout.height = 600

fig.layout.title = 'Total views by cateogry of video'

fig
(top10_world_vid

 .drop_duplicates(subset=['id'])

 .dropna(subset=['id'])

 .assign(video_age_days=lambda df: df['queryTime'].sub(df['snippet.publishedAt']).dt.days)

 .groupby('video_age_days', as_index=False)

 ['statistics.viewCount'].sum()

 .sort_values('statistics.viewCount', ascending=False)

 .reset_index(drop=True)

 .head(10)

 .assign(perc=lambda df: df['statistics.viewCount'].div(df['statistics.viewCount'].sum()).round(2))

 .assign(cum_perc=lambda df: df['perc'].cumsum())

 .style.format({'statistics.viewCount': '{:,}', 'perc': '{:.1%}', 'cum_perc': '{:.1%}'}))
top10_world_vid['video_age_days'] = top10_world_vid['queryTime'].sub(top10_world_vid['snippet.publishedAt']).dt.days

df = top10_world_vid.dropna(subset=['snippet.title']).drop_duplicates('snippet.title')

fig = go.Figure()

fig.add_histogram(x=df['video_age_days'], nbinsx=22)

fig.layout.bargap = 0.1

fig.layout.template = 'none'

fig.layout.title = 'Video age in days'

fig.layout.xaxis.title = 'Age in days'

fig.layout.yaxis.title = 'Number of videos'

fig
tags_stats_cats = (top10_world_vid

                   .drop_duplicates(subset=['snippet.title'])

                   .dropna(subset=['statistics.viewCount'])

                   [['snippet.tags', 'statistics.viewCount', 'category']]

                   .reset_index(drop=True))

tags_stats_cats.head()
tags_stats_cats['snippet.tags'] = [None if pd.isna(x) else eval(x) for x in tags_stats_cats['snippet.tags']]
from collections import defaultdict

dd = defaultdict(lambda: ['', 0])





for i, tag_list in enumerate(tags_stats_cats['snippet.tags']):

    if isinstance(tag_list, list):

        for tag in tag_list:

                dd[tag][0] = tags_stats_cats['category'][i]

                dd[tag][1] += tags_stats_cats['statistics.viewCount'][i]

    else:

        if pd.isna(tag_list):

            dd[None][0] = tags_stats_cats['category'][i]

            dd[None][1] += tags_stats_cats['statistics.viewCount'][i]      



by_tag_category = (pd.DataFrame(list(zip(dd.keys(), dd.values()))).assign(category=lambda df: df[1].str[0],

                                                       view_count=lambda df: df[1].str[1]).drop(columns=[1])

                  .sort_values('view_count', ascending=False)

                  .rename(columns={0: 'tag'})

                  .reset_index(drop=True))

               

by_tag_category.head(10).style.format({'view_count': '{:,}'})
top_fifty_vids = (top10_world_vid

                  .drop_duplicates(subset=['id'])

                  .dropna(subset=['statistics.viewCount'])

                  .sort_values('statistics.viewCount', ascending=False)

                  .head(50))

top_fifty_vids.head(3)
perc_of_views = top_fifty_vids['statistics.viewCount'].sum() / top10_world_vid.drop_duplicates(subset=['id'])['statistics.viewCount'].sum()

num_of_videos = top10_world_vid['id'].nunique()

print(format(perc_of_views, '.2%'), 'of total views were generated by fifty out of', num_of_videos, 'videos (' + format(50/580, '.1%'), 'of the videos).')
import plotly.express as px

fig =  px.treemap(top_fifty_vids.assign(All='All categories'), path=['All', 'category', 'snippet.title'], values='statistics.viewCount')

fig.data[0]['texttemplate'] = '<b>%{label}</b><br><br>Total views: %{value}<br>%{percentParent} of %{parent}'

fig.data[0]['hovertemplate'] = '<b>%{label}</b><br><br>Total views: %{value}<br>%{percentParent:.2%} of %{parent}'

fig.layout.height = 650

fig.layout.title = 'Top 50 trending videos on YouTube by category (65% of total views) - Feb 7, 2020<br>(click on videos and categories to zoom in and out)'

fig.layout.template = 'none'

fig