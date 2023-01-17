import numpy as np

import pandas as pd

import math

from matplotlib import pyplot as plt

import seaborn as sns

sns.set_context('notebook', font_scale=1.5, rc={'line.linewidth': 2.5})



df = pd.read_csv("../input/tarantino.csv")



print("Data loaded!")
df.head(5)
df['movie'].value_counts()
df['movie'] = df['movie'].str.replace('Inglorious Basterds', 'Inglorious Bastards')

df[df['movie'] == 'Inglorious Bastards'].head(5)
meta_df = pd.DataFrame(columns=['release'])

meta_df.loc['Pulp Fiction'] = 'October 14, 1994'

meta_df.loc['Reservoir Dogs'] = 'October 8, 1992'

meta_df.loc['Jackie Brown'] = 'December 25, 1997'

meta_df.loc['Django Unchained'] = 'December 25, 2012'

meta_df.loc['Kill Bill: Vol. 1'] = 'October 10, 2003'

meta_df.loc['Kill Bill: Vol. 2'] = 'April 16, 2004'

meta_df.loc['Inglorious Bastards'] = 'August 21, 2009'



# convert release date column to datetime, then sort ascending

meta_df['release'] = pd.to_datetime(meta_df['release'], infer_datetime_format=True)

meta_df = meta_df.sort_values(by='release')



# title list ordered by release date for plotting

release_order = list(meta_df.index)



meta_df.head(7)
# setting up the palette and figure size

pal = {'word': 'black', 'death': 'red'}

figs = (10,6)

print("Ready to plot")
# stripplot with jitter

plt.figure(figsize=figs)

ax = sns.stripplot(data=df, 

                   x="minutes_in", y="movie", hue="type", palette=pal,

                   order=release_order, jitter=1)
# swarm plot (stripplot with 0 overlap)

plt.figure(figsize=figs)

bx = sns.swarmplot(data=df, x="minutes_in", y="movie", 

                   hue="type", palette=pal, order=release_order)
# bar plot of total counts

plt.figure(figsize=figs)

ax = sns.countplot(data=df, y="movie", 

                   hue="type", palette=pal, order=release_order)
meta_df['duration'] = 0

meta_df.set_value('Pulp Fiction', 'duration', 178)

meta_df.set_value('Reservoir Dogs', 'duration', 99)

meta_df.set_value('Jackie Brown', 'duration', 154)

meta_df.set_value('Django Unchained', 'duration', 165)

meta_df.set_value('Kill Bill: Vol. 1', 'duration', 112)

meta_df.set_value('Kill Bill: Vol. 2', 'duration', 138)

meta_df.set_value('Inglorious Bastards', 'duration', 155)



meta_df.head(7)
df['fgiven'] = df['word'].str.contains('fuck')

df['fgiven'] = df['fgiven'].fillna(0)

df['fgiven'] = df['fgiven'].astype(int)



df[df['fgiven'] == 1].head(10)
fdf = df[df['fgiven'] == 1].copy()

fdf['minutes_in_int'] = fdf['minutes_in'].astype(int)



# bar plot of total counts

plt.figure(figsize=figs)

ax = sns.countplot(data=fdf, y="movie", order=release_order)



# swarm plot colored by colorful language

plt.figure(figsize=figs)

bx = sns.swarmplot(data=fdf, x="minutes_in", y="movie", 

                   hue="word", order=release_order)
def create_fg_df(movie):

    minutes = meta_df.loc[movie]['duration']

    # last minute fucks?

    minutes = range(0, minutes + 1)

    # initiate the fg_df dataframe with column of 0s

    fg_df = pd.DataFrame(index=minutes, columns=['fgiven'])

    fg_df['fgiven'] = 0

    # iterate the minutes of the movie counting fucks

    for minute in minutes:

        fg_in_minute = fdf[fdf['minutes_in_int'] == minute]['fgiven'].sum()

        fg_df.set_value(minute, 'fgiven', fg_in_minute)

    return fg_df



fg_df_rdogs = create_fg_df("Reservoir Dogs")

fg_df_rdogs.head(10)
def rmsfgpm(fg_df):

    fg_df['fgiven'] = fg_df['fgiven']**2

    msfgpm = fg_df['fgiven'].mean()

    return math.sqrt(msfgpm)

print("RMSFGPM for Reservoir Dogs: " + str(rmsfgpm(fg_df_rdogs)))
meta_df['rmsfgpm'] = 0.0

for movie in meta_df.index:

    fg_df = create_fg_df(movie)

    score = rmsfgpm(fg_df)

    meta_df.set_value(movie, 'rmsfgpm', score)



meta_df = meta_df.sort_values(by='rmsfgpm', ascending=False)

meta_df.head(7)
meta_df['fgiven'] = 0

for movie in meta_df.index:

    fs = fdf[fdf['movie'] == movie]['fgiven'].sum()

    meta_df.set_value(movie, 'fgiven', fs)



meta_df = meta_df.sort_values(by='fgiven', ascending=False)

meta_df.head(7)