import numpy as np

import pandas as pd



# Reading the csv file into a DataFrame

df = pd.read_csv('../input/youtube-new/USvideos.csv')

df
df[['video_id', 'title']]
df.loc[:, ['video_id', 'title']]
df.loc[:, ['channel_title']].drop_duplicates()
df.loc[:, ['video_id', 'title']].head(5)
df.loc[:, ['video_id', 'title']].tail(5)
df.loc[:, ['views']].min()
df.loc[:, ['views']].max()
df.loc[:, ['views']].count()
df.loc[:, ['views']].mean()
df.loc[:, ['views']].sum()
new_df = df.loc[:, ['likes']].max().rename({'likes': 'MAX(likes)'})

new_df['MIN(dislikes)'] = df.loc[:, ['dislikes']].min().values[0]

new_df
df.loc[df['likes'] >= 1000000, ['video_id', 'title']]
df.loc[(df['likes'] >= 1000000) & (df['dislikes'] <= 5000), ['video_id', 'title']].drop_duplicates()
df.loc[~pd.isnull(df['description']), ['video_id', 'title']].drop_duplicates()
import re



def like(x, pattern):

    r = re.compile(pattern)

    vlike = np.vectorize(lambda val: bool(r.fullmatch(val)))

    return vlike(x)



df_notnull = df.loc[~pd.isnull(df['description']), :]

df_notnull.loc[like(df_notnull['description'], '.* math .*'), ['video_id', 'title']].drop_duplicates()
df.loc[df['likes'] >= 1000000, ['video_id', 'title']].sort_values(by=['title'], ascending=True).drop_duplicates()
df.loc[:, ['channel_title', 'views', 'likes', 'dislikes']].groupby(['channel_title']).sum()
g = df.groupby(['channel_title'])

g = g.filter(lambda x: x['video_id'].count() > 100)

g = g.loc[:, ['channel_title', 'views', 'likes', 'dislikes']].groupby(['channel_title']).mean()

g
new_row = pd.DataFrame({'video_id': ['EkZGBdY0vlg'],

                        'channel_title': ['Professor Leonard'],

                        'title': ['Calculus 3 Lecture 13.3: Partial Derivatives']})

df = df.append(new_row, ignore_index=True)

df
df.drop(np.where(~(df['channel_title'] == '3Blue1Brown'))[0])
df['like_ratio'] = df['likes'] / (df['likes'] + df['dislikes'])
df
del df['comments_disabled']
df
df.loc[df['channel_title'] == 'Veritasium', ['title', 'likes']]
df['likes'] = np.where(df['channel_title'] == 'Veritasium', df['likes']+100, df['likes'])
df.loc[df['channel_title'] == 'Veritasium', ['title', 'likes']]
df_titles = df.loc[:, ['video_id', 'title']].drop_duplicates()

df_titles
df_stats = df.loc[:, ['video_id', 'views', 'likes', 'dislikes']].groupby('video_id').max()

df_stats = df_stats.reset_index()

df_stats
df_titles.join(df_stats.set_index('video_id'), on='video_id', how='inner')
df_titles.join(df_stats.set_index('video_id'), on='video_id', how='outer')
df_titles.join(df_stats.set_index('video_id'), on='video_id', how='left')
df_titles.join(df_stats.set_index('video_id'), on='video_id', how='right')