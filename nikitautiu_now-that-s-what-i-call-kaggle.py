%matplotlib inline

# pandas

import pandas as pd



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# this styling is purely my preference

# less chartjunk

sns.set_context('notebook', font_scale=1.5, rc={'line.linewidth': 2.5})

sns.set(style='ticks', palette='Set2')
# loading the data

df = pd.read_csv('../input/Now_with_Spotify_Final.csv')

df.head()
release_years = [1998, 1999, 1999, 2000, 2000, 2001, 2001, 2001, 2002, 2002, 2002, 2003, 2003, 2003,

                 2004, 2004, 2004, 2005, 2005, 2005, 2006, 2006, 2006, 2007, 2007, 2007, 2008, 2008, 

                 2008, 2009, 2009, 2009, 2010, 2010, 2010, 2010, 2011, 2011, 2011, 2011, 2012, 2012, 

                 2012, 2012, 2013, 2013, 2013, 2013, 2014, 2014, 2014, 2014, 2015, 2015, 2015, 2015, 

                 2016, 2016, 2016, 2016, 2017]

df['year'] = df['volume_number'].apply(lambda x: release_years[x - 1])  # 0 indexed



fig = plt.figure(figsize=(16,4))

plt.subplot(121)

sns.countplot(data=df, x='year', color=sns.color_palette('Set2')[0])

plt.xticks(rotation=45)

plt.title('number of tracks')

sns.despine()





plt.subplot(122)

sns.countplot(data=df.groupby('volume_number').first(), x='year', color=sns.color_palette('Set2')[0])

plt.xticks(rotation=45)

plt.title('number of albums')

sns.despine()
# split the artists into mains and features

artists = df['artist'].str.split(r'\s*featuring\s*').apply(pd.Series)



# get the artists

main_artist = artists[0].str.split(r'(\s+and\s+)|,').apply(pd.Series).stack()

main_df = pd.DataFrame(data={'artist_name': main_artist, 'featuring': False})

main_df.index = main_df.index.droplevel(-1)



# get the featuring artists

feat_artist = artists[1].dropna().str.split(r'\s+and\s+|,').apply(pd.Series).stack()

feat_df = pd.DataFrame(data={'artist_name': feat_artist, 'featuring': True})

feat_df.index = feat_df.index.droplevel(-1)



# join the results

artist_df = pd.concat([df.join(main_df, how='inner'), df.join(feat_df, how='inner')], axis='rows')



# strip the spaces and drop the redundant 'and's

artist_df['artist_name'] = artist_df['artist_name'].str.strip()

artist_df = artist_df[artist_df['artist_name'] != 'and']  #  split fails sometimes
# group by song and 

feat_stats = artist_df.groupby([artist_df.index, 'featuring'], ).agg({'title': ['count', 'first'], 'artist': 'first', 'year': 'first'})

feat_stats = feat_stats.unstack().T.reset_index(drop=True).T

feat_stats = feat_stats.drop([3, 5, 7], axis='columns').fillna(0)  # some songs just don't have any features

feat_stats.columns = ['main', 'feat', 'title', 'artist', 'year']

feat_stats['total'] = feat_stats['main'] + feat_stats['feat']

df = df.join(feat_stats[['main', 'feat', 'total']])  #  add them to the input dataframe
feat_stats[['main', 'feat', 'total']].apply(pd.value_counts, axis='rows').fillna(0)
feat_stats.sort_values('total', ascending=False).head(7)
plt.figure(figsize=(8,5))

g = sns.pointplot(data=df[['year', 'total']], y='total', x='year', join=False)

plt.xticks(rotation=45)

sns.despine()
# number of personal songs and collaborations

artist_stats = artist_df.groupby('artist_name').apply(lambda grp: pd.Series({

    'own_songs': (~grp['featuring']).sum(),

    'features': grp['featuring'].sum()

}))

artist_stats['total_songs'] = artist_stats['own_songs'] + artist_stats['features']

artist_stats['featurability'] = artist_stats['features'] / artist_stats['total_songs']



# add musical metric stats

metric_cols = ['speechiness', 'key', 'time_signature', 'liveness', 'loudness',

    'duration_ms', 'danceability', 'duration', 'valence', 'acousticness',

    'spotify_id', 'volume_number', 'energy', 'tempo', 'instrumentalness',

    'mode']



# we just use the mean for these ones

artist_music_stats = artist_df.groupby('artist_name')[metric_cols].mean()

artist_stats = artist_stats.join(artist_music_stats, how='left')
# data needs to be wrangled a bit to use it with seaborns grids

plot_df = (artist_stats[['own_songs', 'features', 'total_songs']]

    .stack()

    .reset_index()

    .rename(columns={'level_1': 'variable', 0: 'value'}))



# plot it sharex is disabled so the xlabels aren't locked ogether

g = sns.FacetGrid(data=plot_df, col='variable', size=5, sharex=False)

g.map(sns.countplot, 'value');
plt.figure(figsize=(10, 5))

sns.distplot(artist_stats['featurability'], hist=True, rug=True, kde=False)

plt.xlim(0, 1)

sns.despine()
artist_stats['total_songs'].sort_values(ascending=False).head(20)
artist_stats['own_songs'].sort_values(ascending=False).head(20)
artist_stats['featurability'].sort_values(ascending=False).head(10)