#!pip install pysqlite3 --upgrade
# Imports and connecting to the database

import sqlite3

import pandas as pd

import seaborn as sns

import numpy as np 

import pylab 

import scipy.stats as stats

import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('seaborn')

%matplotlib inline



sqlite_file = '../input/database.sqlite'

conn = sqlite3.connect(sqlite_file)
# Query for a summary dataframe

q = """SELECT artists.reviewid,artists.artist,title,score,genre,year,best_new_music FROM artists 

                    INNER JOIN genres ON artists.reviewid=genres.reviewid 

                    INNER JOIN years on artists.reviewid=years.reviewid 

                    INNER JOIN reviews ON artists.reviewid=reviews.reviewid"""

df = pd.read_sql_query(q, conn)

df.info()
# Clean up some

df.dropna(axis=0,how='any',inplace=True)

df.reset_index(inplace=True)

df.drop(labels=['index'],axis=1,inplace=True)

#df.info()

#df.describe()

df.head()
# Count number of reviews by genre

reviews_artist = df.groupby('artist').count().reset_index().rename(columns={'genre':'reviews'})[['artist','reviews']].set_index('artist').sort_values('reviews',ascending=False)

reviews_genre = df.groupby('genre').count().reset_index().rename(columns={'artist':'reviews'})[['genre','reviews']].set_index('genre').sort_values('reviews',ascending=False)

print(reviews_genre)

top_genres = reviews_genre.index[:5]



# Plot the data

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn')

reviews_genre.plot(kind='bar',fontsize=14,title='Number of Reviews')

#reviews_artist[:10].plot(kind='bar',fontsize=14)
# Visualize number of reviews by genre over time

top_genres_boolean_mask = []

for i in range(len(df)):

    top_genres_boolean_mask.append(df['genre'][i] in top_genres)

reviews_year_genre = df[(df['year']>=2000) & (df['year']<2017) & top_genres_boolean_mask].groupby(['year','genre']).count().drop(labels='reviewid',axis=1).rename(columns={'artist':'reviews'}).sort_values(['year','reviews'],ascending=False)['reviews']

reviews_year_genre.unstack().plot(kind='line',fontsize=14,title="Number of Reviews")
# Visualize mean & std dev of scores by genre over time

top_genres_boolean_mask = []

for i in range(len(df)):

    top_genres_boolean_mask.append(df['genre'][i] in top_genres)

top_genres_boolean_mask = (df['year']>=2000) & (df['year']<2017) & top_genres_boolean_mask



reviews_score_mean = df[top_genres_boolean_mask].groupby(['year','genre']).mean().drop(labels='reviewid',axis=1)[['score']].sort_values(['year','score'],ascending=False)

reviews_score_std = df[top_genres_boolean_mask].groupby(['year','genre']).std().drop(labels='reviewid',axis=1)[['score']].sort_values(['year','score'],ascending=False)

reviews_score_mean.unstack().plot(kind='line',fontsize=14,title='Mean Scores')

reviews_score_std.unstack().plot(kind='line',fontsize=14,title='Std Scores')
# Visualize frequency of best new music by genre

top_genres_boolean_mask = []

for i in range(len(df)):

    top_genres_boolean_mask.append(df['genre'][i] in top_genres)

top_genres_boolean_mask = (df['year']>=2000) & (df['year']<2017) & top_genres_boolean_mask

bnm_freq = df[top_genres_boolean_mask].groupby(['year','genre']).sum()['best_new_music']/df.groupby('genre').count()['best_new_music']

bnm_freq.unstack().plot(kind='line',fontsize=14,title='Best New Music Frequency')
# Scatter plot of mean artist score vs best new music freq

grp_artist = df.groupby('artist').mean()

sns.jointplot("score", "best_new_music", data=grp_artist, kind="reg")
# Are scores normally distributed??

stats.probplot(df['score'], dist="norm", plot=pylab)

pylab.show()
# Score distribution (total & by artists)

df['score'].hist()

grp_artist['score'].hist()
# Create Dummy Variable columns for linear regression

df_dummies = pd.get_dummies(data=df,columns=['genre'],drop_first=True)

df_dummies.columns
# Can score be predicted from year and genre?

import statsmodels.api as sm

x = df_dummies[['year','genre_experimental', 'genre_folk/country', 'genre_global',

       'genre_jazz', 'genre_metal', 'genre_pop/r&b', 'genre_rap',

       'genre_rock']]

y = df_dummies['score']

model = sm.OLS(y,x).fit()

model.summary()
# A slimmer model only including significant predictors

import statsmodels.api as sm

x = df_dummies[['year','genre_experimental', 'genre_folk/country', 'genre_global',

       'genre_jazz','genre_rock']]

y = df_dummies['score']

model_2= sm.OLS(y,x).fit()

model_2.summary()
# Returns the average scores of a list of arists, weighted equally

# Expand to give full artist summary

def score_artists(artists_list):

    cum_score = 0

    for artist in artists_list:

        cum_score += df[df['artist']==artist]['score'].mean()

    return cum_score/len(artists_list)



my_artists = ['david bowie','prince','led zeppelin']

pop_artists = ['ariana grande','sza','justin bieber']

print(score_artists(my_artists),score_artists(pop_artists))
# Returns a list of the top n artists in a genre by mean album score or best new music freq

# Genres are 'electronic', 'metal', 'rock', 'rap', 'experimental', 'pop/r&b','folk/country', 'jazz', 'global'

# Method is either 'best', 'score', or 'both'

def top_genre_artists(genre,method='both',n=5,min_albums=3,year=2000):

    if genre not in df['genre'].unique():

        print('Invalid genre')

        return []

    elif method not in ['best','score','both']:

        print('Invalid method')

        return []

    print('The top '+str(n)+' artists in '+genre+' are:')

    if method == 'score':

        print('(Ranked by score, '+str(min_albums)+' album minimum)')

        artist_grouped = df[(df['genre']==genre) & (df['year']>=year)].groupby('artist')

        return artist_grouped.mean()[artist_grouped.size()>=min_albums]['score'].sort_values(ascending=False)[:n]

    if method == 'best':

        print('(Ranked by best new music frequency, '+str(min_albums)+' album minimum)')

        artist_grouped = df[(df['genre']==genre) & (df['year']>=year)].groupby('artist')

        return artist_grouped.mean()[artist_grouped.size()>=min_albums]['best_new_music'].sort_values(ascending=False)[:n]

    if method == 'both':

        print('(Ranked by score * best new music frequency, '+str(min_albums)+' album minimum)')

        artist_grouped = df[(df['genre']==genre) & (df['year']>=year)].groupby('artist')

        score = artist_grouped.mean()[artist_grouped.size()>=min_albums]['score']

        best = artist_grouped.mean()[artist_grouped.size()>=min_albums]['best_new_music']

        return (best*score).sort_values(ascending=False)[:n]



#top_genre_artists('electronic','score')

top_genre_artists('rock')



# THERE IS A SLIGHT PROBLEM WITH THIS METHOD BECAUSE REISSUED ALBUM REVIEWS APPEAR TWICE

#   -> DISTORTING THE MIN ALBUM REQUIREMENT, WILL TRY TO FIND A SOLUTION

# MULTI-GENRE REVIEWS ALSO APPEAR TWICE BUT ARE FILTERED SO NO PROBLEM
df['year'].hist()

# Will explore further...