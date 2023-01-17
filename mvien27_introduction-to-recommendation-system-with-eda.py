# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import sys
from itertools import combinations, groupby
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
metadata = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', parse_dates=True)
metadata.head()
metadata.genres = [', '.join(i for i in re.findall(r"(?<='name': ')\w+(?='})", j)) for j in metadata.genres]

def freq(column, xlab, ylab, title): 
    str = ', '.join(i.replace(',','') for i in column).replace(',', '')
    str = str.split()          
    str2 = []
    count = []
    for i in str:              
        if i not in str2:
            str2.append(i)    
    for i in range(0, len(str2)):
        count.append(str.count(str2[i]))
        print('Frequency of', str2[i], 'is :', str.count(str2[i]))
        
    df = pd.DataFrame({'word': str2, 'frequency': count}).sort_values(ascending=False, by=['frequency'])[::-1]
    # Create trace
    trace = go.Bar(x = df.frequency, text = df['word'],
                   textposition = 'outside', textfont = dict(color = '#000000'),
                   orientation = 'h', y = list(range(0, len(str2))), marker = dict(color = '#db0000'))
    # Create layout
    layout = dict(title = title, xaxis = dict(title = xlab, range = (0, max(count)+2000)),
                  yaxis = dict(title = ylab))
    # Create plot
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
          
freq(metadata.genres, xlab='frequency', ylab='genres', title='Genres Frequency')
metadata.spoken_languages = [', '.join(i for i in re.findall(r"(?<='name': ')[a-zA-Z\s]+(?='})", str(j))) for j \
                                 in metadata.spoken_languages]

languages = metadata.spoken_languages.value_counts().sort_values(ascending=False)[:10][::-1]

trace = go.Bar(x = languages.values, text = languages.values, orientation = 'h',
                   textposition = 'auto', textfont = dict(color = '#000000'),
               y = languages.index, marker = dict(color = '#db0000'))
    
layout = dict(title = 'Top 20 Most Spoken Languages', xaxis = dict(title = 'Count'),
              yaxis = dict(title = 'Languages'))
    
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
metadata.production_companies = [', '.join(i for i in re.findall(r"(?<={'name': ')[a-zA-Z\s]+(?=',)", str(j))) for j \
                                 in metadata.production_companies]

# Remove 1st value because there is no company shown
comps = metadata.production_companies.value_counts().sort_values(ascending=False)[1:20]

trace = go.Bar(x = comps.index, text = comps.values,
                   textposition = 'auto', textfont = dict(color = '#000000'),
               y = comps.values, marker = dict(color = '#db0000'))
    
layout = dict(title = 'Top 20 production companies', xaxis = dict(title = 'Companies'),
              yaxis = dict(title = 'Count'))
    
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
date = []
for i in metadata[~metadata.release_date.isna()]['release_date']:
    try:
        date.append(pd.to_datetime(i, format='%Y-%m-%d').year)
    except:
        pass
year = pd.Series(date).value_counts().sort_index()

trace = go.Scatter(x = year.index, y = year.values)
# Create layout
layout = dict(title = 'Number of Movies Released Throughout {} Years'.format(year.shape[0]),
              xaxis = dict(title = 'Release Year'),
              yaxis = dict(title = 'Number of Movies'))

# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')
ratings.head()
rate_year = pd.to_datetime(ratings['timestamp'], unit='s').dt.year.value_counts().sort_index()

trace = go.Scatter(x = rate_year.index, y = rate_year.values)
# Create layout
layout = dict(title = 'Number of Movies Ratings Throughout {} Years'.format(rate_year.shape[0]),
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = 'Number of Movies Ratings'))

# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
r = ratings.rating.value_counts().sort_index()

trace = go.Bar(x = r.index, text = ['{:.1f} %'.format(val) for val in (r.values * 100 / r.values.sum())],
                                    y = r.values, textposition = 'auto',
              marker = dict(color = '#df1447'))
# Create layout
layout = dict(title = 'Ratings by Users',
              xaxis = dict(title = 'Ratings'),
              yaxis = dict(title = 'count'))

# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
links = pd.read_csv('../input/the-movies-dataset/links_small.csv')
links.dropna(inplace=True)
links.tmdbId = links.tmdbId.map(int)
links.head()
df = metadata[['genres', 'id', 'imdb_id', 'overview', 'release_date', 'title']]
df.dropna(inplace=True)
df.imdb_id = df.imdb_id.str.replace('tt', '')
df.imdb_id = df.imdb_id.map(int)
df.id = df.id.map(int)
print(df.info())
df.head()
data = ratings.set_index('movieId').join(links.set_index('movieId')).reset_index().set_index('tmdbId').join(df.set_index('id')).dropna()
data = data.reset_index(drop=True)
data.drop(columns=['imdb_id'], inplace=True)
data.sample(6)
data.shape
def filtered_items(n, f, group, series, df):
    stats = df.groupby(group)[series].agg(f)
    stats.index = stats.index.map(int)
    benchmark = round(stats['count'].quantile(n), 0)
    rm_items_index = stats[stats['count'] < benchmark].index
    return rm_items_index
drop_m_ind = filtered_items(n=0.7, f=['count', 'mean'], group='movieId', series='rating', df=ratings)
drop_u_ind = filtered_items(n=0.7, f=['count', 'mean'], group='userId', series='rating', df=ratings)

ratings_filtered = ratings[~ratings.movieId.isin(drop_m_ind)]
ratings_filtered = ratings_filtered[~ratings_filtered.userId.isin(drop_u_ind)].drop('timestamp', axis=1)
ratings_filtered.tail()
n = 20
most_rated = data.groupby('title')['rating'].count().sort_values(ascending=False).head(n)

trace = go.Bar(x = most_rated.index, text = most_rated.values,
                                    y = most_rated.values, textposition = 'auto', marker = dict(color = '#df0003'))

# Create layout
layout = dict(title = 'Top {} Most Rated Movies'.format(n),
              xaxis = dict(title = 'Movie Titles'),
              yaxis = dict(title = 'count'))

# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
movies_link = data[['movieId', 'title', 'genres']].drop_duplicates(['movieId', 'title', 'genres']).set_index('movieId')
movies_link.head()
m = ratings_filtered.pivot_table(index='userId', columns='movieId', values='rating')
user_m = m.fillna(0)
movie_m = user_m.T
m.head()
m.shape
m.count(axis=0)
n = 20

rating_mean = m.mean(axis=0).rename('rating_mean').to_frame()

rating_count = m.count(axis=0).rename('rating_count').to_frame()

d = rating_mean.join(rating_count).join(data[['movieId', 'title']].drop_duplicates\
                                        (subset=['movieId', 'title']).set_index('movieId'))

d = d.sort_values(by='rating_mean', ascending=False)[:n]

print(d)

trace = go.Bar(x = d.title, text = d.rating_mean,
                                    y = d.rating_mean, textposition = 'auto', marker = dict(color = '#df0003'))

# Create layout
layout = dict(title = 'Top {} high rated movies'.format(n),
              xaxis = dict(title = 'Movie Titles'),
              yaxis = dict(title = 'count'))

# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
def cosine_re_userBased(n, userId):
    
    cosine_user_m = cosine_similarity(user_m) - np.eye(user_m.shape[0])
    
    sim_index_user = np.argsort(cosine_user_m[userId-1])[::-1]
    sim_score_user = np.sort(cosine_user_m[userId-1])[::-1]
    
    urated_movies = m.loc[userId,][m.loc[userId,].isna()].index
    
    mean_mov_re = (user_m.iloc[sim_index_user[:50]].T * sim_score_user[:50]).T.mean(axis=0)
    
    top_recommendation = mean_mov_re[urated_movies].sort_values(ascending=False)[:n].to_frame().join(movies_link[['title']])
    
    print(top_recommendation)
    
    trace = go.Bar(x=top_recommendation.title, text=[round(i, 3) for i in top_recommendation.iloc[:, 0]],
                   y=top_recommendation.iloc[:, 0],
                  textposition = 'auto', marker = dict(color = '#df0003'))
    layout = dict(title = 'Top 10 Recommended Movies for User {}'.format(userId),
              xaxis = dict(title = 'Movie Titles'),
              yaxis = dict(title = 'Similarity Score'))
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
cosine_re_userBased(n=20, userId=119)
# similar movies based on users rating and genres
def moviesBased_rec(n, title):
    
    positionId = [movie_m.index.get_loc(i) for i in movies_link[movies_link.title == title].index]
    cosine_movies_m = cosine_similarity(movie_m)
    sim_index_mov = [np.argsort(cosine_movies_m[pId])[::-1][:50] for pId in positionId]
    
    for sim_index_ in sim_index_mov:
        simMovies = (movie_m.iloc[sim_index_,]).T.mean\
        (axis=0).sort_values(ascending=False).to_frame().join(movies_link)
        simMovies = simMovies.reset_index().rename(columns={'index': 'movieId'})
        
        # improve model by involving genres
        guess = []
        for i, v in enumerate(simMovies.genres):
            try:
                for j in v.split(', '):
                    if j in ", ".join(i for i in movies_link[movies_link.title == title]['genres'].values).split(', ') \
                    and j not in guess:
                        guess.append(i)
            except: pass
    
        simMovies = simMovies.iloc[guess,:][:n]
        simMovies = simMovies.drop_duplicates(['movieId', 'title', 'genres'])#[1:]
        print(simMovies.to_string())
        
        trace = go.Bar(x= simMovies.title, text=[round(i, 4) for i in  simMovies.iloc[:, 1]],
                       y= simMovies.iloc[:, 1],
                      textposition = 'auto', marker = dict(color = '#df0003'))
        layout = dict(title = 'Top Similar Rated Movies to {}'.format(title),
                  xaxis = dict(title = 'Movie Titles'),
                  yaxis = dict(title = 'Similarity Score'))
        fig = go.Figure(data=[trace], layout=layout)
        iplot(fig)
moviesBased_rec(n=20, title='The Notebook')
def may_also_like(n, title):
    # get the movieId
    movieId = [i for i in movies_link[movies_link.title == title].index]
    
    # begin the loop
    for mId in movieId:
        # users that watched the movies
        watched_users = [uid for uid in m.loc[:, mId].dropna().index]
        # get the Id of users that rated more than 3 for the movie
        ref_users = [index for (index, value) in zip(watched_users, \
                                                     m.loc[watched_users, mId]) if value >=3.0]
        # create matrix with columns of ref_users 
        sameWatchs = m.loc[ref_users,:].dropna(how='all', axis=1)
        sameWatchs = sameWatchs.T
        #sameWatchs_m = sameWatchs.T.fillna(sameWatchs.mean(axis=1)).T
        sameWatchs_m = sameWatchs.fillna(0)
        
        # get the index of movieId inside matrix of ref_users
        pos = sameWatchs.index.get_loc(sameWatchs.loc[mId,:].name)
    
    
        cosine_ref_users = (cosine_similarity(sameWatchs_m) - np.eye(sameWatchs_m.shape[0]))
        # get the 50 highest cosine scores
        sim_index_mov = np.argsort(cosine_ref_users[pos])[::-1][:50]
    
        # calculate the mean of 50 movie with highest score from ref_users matrix 
        posibility = (sameWatchs_m.iloc[sim_index_mov,:]).T.mean\
                     (axis=0).sort_values(ascending=False).to_frame().join(movies_link)
        posibility = posibility.reset_index().rename(columns={'index': 'movieId'})[:n]
        
        guess = []
        for i, v in enumerate(posibility.genres):
            for j in v.split(', '):
                if j in ", ".join(i for i in movies_link[movies_link.title == title]['genres'].values).split(', ') \
                and j not in guess:
                    guess.append(i)
        guess = list(set([x for x in guess if guess.count(x) >= 1]))
        final = posibility.iloc[guess,:][:n]
        final = final.drop_duplicates(['movieId', 'title', 'genres'])
        print(final.to_string())
    
        trace = go.Bar(x= final.title, text=[round(i, 4) for i in  final.iloc[:, 1]],
                        y= final.iloc[:, 1],
                        textposition = 'auto', marker = dict(color = '#df0003'))
        layout = dict(title = 'Top {} Recommended Movies if you enjoy {}'.format(n, title),
                      xaxis = dict(title = 'Movie Titles'),
                      yaxis = dict(title = 'Similarity Score'))
        fig = go.Figure(data=[trace], layout=layout)
        iplot(fig)
may_also_like(n=20, title='The Notebook')
def next_movies(n, userId):
    
    watched = data[(data.userId == userId) & (data.rating >= 4)]['movieId']
    
    #position = np.argmax(cosine_sim(user_m) - np.eye(user_m.shape[0]), axis=1).tolist()[userId-1]
    position = np.argmax(cosine_similarity(user_m) - np.eye(user_m.shape[0]), axis=1).tolist()[userId-1]

    #suggest = user_m.iloc[position,].sort_values(ascending=False).to_frame().join(movies_link)
    suggest = m.fillna(0).iloc[position,].sort_values(ascending=False).to_frame().join(movies_link)
    
    for w in watched:
        if w in suggest.index: suggest = suggest.drop(w, axis=0)
        else: pass
    suggest = suggest[:n]
    
    print(suggest.to_string())
    trace = go.Bar(x= suggest.title, text=[round(i, 4) for i in  suggest.iloc[:, 0]],
                   y= suggest.iloc[:, 0],
                  textposition = 'auto', marker = dict(color = '#df0003'))
    layout = dict(title = 'Top {} Recommended Movies for User {}'.format(n, userId),
              xaxis = dict(title = 'Movie Titles'),
              yaxis = dict(title = 'Similarity Score'))
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
next_movies(n=20, userId=119)
# Getting suggestions from users of same ratings

def next_movies2(n, userId):

    bestMovies = [i for i, v in zip(data[data.userId == userId]['movieId'],
                                    data[data.userId == userId]['rating']) if v >= 4]
    sameWatchs = m.loc[:, bestMovies].dropna(how='all').index

    pref = []
    for ui in sameWatchs:
        for mId in bestMovies:
            try:
                if m.loc[ui, mId] == np.nan: pass
                else:
                    if m.loc[ui, mId] >= 4:
                        pref.append(ui)
            except: pass
    following_suggest = list(set([x for x in pref if pref.count(x) >= 2]))
    #following_suggest.remove(userId)
    following_m = user_m.loc[following_suggest,]
    
    
    watched = data[(data.userId == userId) & (data.rating >= 4)]['movieId']
    
    position = following_m.index.get_loc(userId)

    suggest = following_m.iloc[np.argmax(cosine_similarity(following_m) - np.eye(following_m.shape[0]), axis=1).tolist()\
                               [position]][::-1].sort_values(ascending=False).to_frame().join(movies_link)
    for w in watched:
        if w in suggest.index: suggest = suggest.drop(w, axis=0)
        else: pass
    suggest = suggest[:n]
    
    print(suggest.to_string())
    trace = go.Bar(x= suggest.title, text=[round(i, 4) for i in  suggest.iloc[:, 0]],
                   y= suggest.iloc[:, 0],
                  textposition = 'auto', marker = dict(color = '#df0003'))
    layout = dict(title = 'Top {} Recommended Movies for User {}'.format(n, userId),
              xaxis = dict(title = 'Movie Titles'),
              yaxis = dict(title = 'Similarity Score'))
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
next_movies2(n=20, userId=119)
m_corr = ratings_filtered.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
m_corr.head()
def pearson_rec(n, title):
    print("- Top 10 movies recommended for {} based on Pearsons'R correlation - ".format(title))
    # mId = int(movies_link2.index[movies_link2['title'] == title][0])
    mId = [i for i in movies_link[movies_link['title'] == title].index]
    targets = [user_m.loc[:,i] for i in mId]
    sim_targets = [user_m.corrwith(target) for target in targets]
    for sim_target in sim_targets:
        corr_target = pd.DataFrame(sim_target, columns = ['PearsonR'])
        corr_target.dropna(inplace = True)
        corr_target = corr_target.sort_values('PearsonR', ascending = False)
        corr_target.index = corr_target.index.map(int)
        corr_target = corr_target.join(movies_link)[['PearsonR', 'title', 'genres']][:n]
        print(corr_target.to_string(index=False))
    
        # Graph
        trace = go.Bar(x=corr_target.title, text=[round(i, 4) for i in  corr_target['PearsonR']],
                        y= corr_target['PearsonR'],
                        textposition = 'auto', marker = dict(color = '#df0003'))
        layout = dict(title = 'Top Similar Movies to {}'.format(title),
                      xaxis = dict(title = 'Movie Titles'),
                      yaxis = dict(title = 'PearsonR Score'))
        fig = go.Figure(data=[trace], layout=layout)
        iplot(fig)
    
pearson_rec(n = 20, title='Batman Begins')
pearson_rec(n = 20, title='Iron Man')
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors

def fuzzy_match(title):
    match = [fuzz.ratio(title, i) for i in data['title']]
    return data.iloc[match.index(100),]['movieId']

def Knn_recommendation(n, data, matrix, title):
    
    csr_m_knn = csr_matrix(matrix)
    #make an object for the NearestNeighbors Class.
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n+1, n_jobs=-1)

    # fit the dataset
    model_knn.fit(csr_m_knn)

    ix = fuzzy_match(title=title)
    idx = matrix.index.get_loc(matrix.loc[ix,].name)

    distances, indices = model_knn.kneighbors(csr_m_knn[idx], n_neighbors=n+1)

    df = pd.DataFrame({'position': indices.squeeze(), 'score': distances.squeeze()}).sort_values\
    (by='score', ascending=False)

    movies_title = [data[data.movieId == matrix.iloc[i,].name]['title'].values[0] for i in df['position']]

    df['title'] = movies_title
    print(df.to_string())
    
    trace = go.Bar(x= df.title, text=[round(i, 4) for i in  df['score']],
                    y=df['score'], textposition = 'auto', marker = dict(color = '#df0003'))
    layout = dict(title = 'Top Similar Rated Movies to {}'.format(title),
                  xaxis = dict(title = 'Movie Titles'),
                  yaxis = dict(title = 'Similarity Score'))
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
Knn_recommendation(n=20, data=data, matrix=movie_m, title='Iron Man')
Knn_recommendation(n=20, data=data, matrix=movie_m, title='Batman Begins')
Knn_recommendation(n=20, data=data, matrix=movie_m, title='The Notebook')
def pearson_rec(n, title):
    print("- Top 10 movies recommended for {} based on Pearsons'R correlation - ".format(title))
    # mId = int(movies_link2.index[movies_link2['title'] == title][0])
    mId = [i for i in movies_link[movies_link['title'] == title].index]
    targets = [user_m.loc[:,i] for i in mId]
    sim_targets = [user_m.corrwith(target) for target in targets]
    for sim_target in sim_targets:
        corr_target = pd.DataFrame(sim_target, columns = ['PearsonR'])
        corr_target.dropna(inplace = True)
        corr_target = corr_target.sort_values('PearsonR', ascending = False)
        corr_target.index = corr_target.index.map(int)
        corr_target = corr_target.join(movies_link)[['PearsonR', 'title', 'genres']][:n]
        print(corr_target.to_string(index=False))
    
        # Graph
        trace = go.Bar(x=corr_target.title, text=[round(i, 4) for i in  corr_target['PearsonR']],
                        y= corr_target['PearsonR'],
                        textposition = 'auto', marker = dict(color = '#df0003'))
        layout = dict(title = 'Top Similar Movies to {}'.format(title),
                      xaxis = dict(title = 'Movie Titles'),
                      yaxis = dict(title = 'PearsonR Score'))
        fig = go.Figure(data=[trace], layout=layout)
        iplot(fig)
    
pearson_rec(n = 20, title='Batman Begins')
pearson_rec(n = 20, title='Iron Man')
from scipy.sparse.linalg import svds

m_demeaned = user_m - np.mean(user_m, axis=1).values.reshape(-1, 1)

U, sigma, Vt = svds(m_demeaned, k = 50)
sigma = np.diag(sigma)

print(m_demeaned.shape)
print(U.shape)
print(sigma.shape)
print(Vt.shape)
user_preds = np.dot(np.dot(U, sigma), Vt) + np.mean(user_m, axis=1).values.reshape(-1, 1)
preds_df = pd.DataFrame(user_preds, columns = user_m.columns)
preds_df.index = user_m.index
preds_df.head()
def svd_recommendation(n, user, user_predictions):
    # sort the prediction df
    sorted_preds = user_predictions.loc[user,].sort_values(ascending=False)
    
    # 
    watched = data[data.userId == user][['userId', 'rating', 'movieId', 'title']]
    unwatched = movies_link[~movies_link.index.isin(watched['movieId'])]
    
    recommendations = unwatched.join(sorted_preds).rename(columns={user: 'pred'}).sort_values(by='pred', ascending=False)[:n]
    
    # Graph
    trace = go.Bar(x= recommendations.title, text=[round(i, 4) for i in  recommendations['pred']],
                    y=recommendations['pred'], textposition = 'auto', marker = dict(color = '#df0003'))
    layout = dict(title = 'Top Recommended Movies to user {}'.format(user),
                  xaxis = dict(title = 'Movie Titles'),
                  yaxis = dict(title = 'Prediction Score'))
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
svd_recommendation(n=20, user=119, user_predictions=preds_df)
m_m_demeaned = movie_m - np.mean(movie_m, axis=1).values.reshape(-1, 1)
U_m, sigma_m, Vt_m = svds(m_m_demeaned, k = 50)
sigma_m = np.diag(sigma_m)
movie_preds = np.dot(np.dot(U_m, sigma_m), Vt_m) + np.mean(movie_m, axis=1).values.reshape(-1, 1)
preds_m_df = pd.DataFrame(movie_preds, columns = movie_m.columns)
preds_m_df.index = movie_m.index
preds_m_df.head()
def svd_movies_recommendation(n, title):
    
    positionId = [preds_m_df.index.get_loc(i) for i in movies_link[movies_link.title == title].index]
    cosine_movies_m = cosine_similarity(preds_m_df)
    sim_index_mov = [np.argsort(cosine_movies_m[pId])[::-1][:50] for pId in positionId]
    
    for sim_index_ in sim_index_mov:
        simMovies = (preds_m_df.iloc[sim_index_,]).T.mean\
        (axis=0).sort_values(ascending=False).to_frame().join(movies_link)
        simMovies = simMovies.rename(columns={0: 'pred'})[:n]
    
        print(simMovies.to_string())
        
        trace = go.Bar(x= simMovies.title, text=[round(i, 4) for i in  simMovies['pred']],
                       y= simMovies['pred'],
                      textposition = 'auto', marker = dict(color = '#df0003'))
        layout = dict(title = 'Top Similar Rated Movies to {}'.format(title),
                  xaxis = dict(title = 'Movie Titles'),
                  yaxis = dict(title = 'Similarity Score'))
        fig = go.Figure(data=[trace], layout=layout)
        iplot(fig)
        
svd_movies_recommendation(n=20, title='The Notebook')
# Define a function for calculating pair of items

def get_item_pairs(order_item):
    order_item = order_item.reset_index().as_matrix()
    for order_id, order_object in groupby(order_item, lambda x: x[0]):
        item_list = [item[1] for item in order_object]
              
        for item_pair in combinations(item_list, 2):
            yield item_pair

def top_association(n, min_support, min_size):
    
    orders = ratings.drop(columns=['rating', 'timestamp'])
    orders = orders.set_index('userId')['movieId']

    # Calculate frequency and support
    stats = orders.value_counts().to_frame("freq")
    stats['support']  = stats['freq'] / len(set(orders.index))
    
    # Filter out items below min support 
    qualifying_items = stats[stats['support'] >= min_support].index
    orders = orders[orders.isin(qualifying_items)]

    # Filter out orders with less than minimum requirement
    order_size = orders.index.value_counts().rename("freq")
    qualifying_orders  = order_size[order_size >= min_size].index
    orders  = orders[orders.index.isin(qualifying_orders)]
    
    # Recalculate item frequency and support
    stats  = orders.value_counts().rename('freq').to_frame("freq")
    stats['support']  = stats['freq'] / len(set(orders.index))

    # get values for pair of items
    pair_gen  = get_item_pairs(orders)
    
    # Calculate item pair frequency and support
    pairs  = pd.Series(Counter(pair_gen)).rename('freq').to_frame("freqAB")
    pairs['supportAB'] = pairs['freqAB'] / len(qualifying_orders)
    
    # Filter from item_pairs those below min support
    pairs = pairs[pairs['supportAB'] >= min_support]

    # Create table of association rules and compute relevant metrics
    pairs = pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    pairs = pairs\
    .merge(stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)\
    .merge(stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True)

    # Create table of association rules
    pairs['confidenceAtoB'] = pairs['supportAB'] / pairs['supportA']
    pairs['confidenceBtoA'] = pairs['supportAB'] / pairs['supportB']
    pairs['lift']           = pairs['supportAB'] / (pairs['supportA'] * pairs['supportB'])
    
    # sort dataframe by lift
    rules = pairs.sort_values('lift', ascending=False)
    
    # merge with movie title dataframe
    title_list = movies_link.drop('genres', axis=1).reset_index()

    final_rules = rules.merge(title_list.rename(columns={'title': 'itemA'}), left_on='item_A', right_on=\
                              'movieId').merge(title_list.rename(columns={'title': 'itemB'}),
                                               left_on='item_B', right_on='movieId').sort_values('lift', ascending=False)

    final_rules = final_rules[['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 
                               'confidenceAtoB','confidenceBtoA','lift']]
    
    # Remove instances with matched titles
    final_rules = final_rules[final_rules.itemA != final_rules.itemB]
    
    # Graph
    trace = go.Bar(x=list(range(1, n+1)), text=final_rules.itemA[:n].astype(str)+" & "+final_rules.itemB[:n].astype(str),
                    y=final_rules.lift[:n], textposition = 'auto', marker = dict(color = '#df0003'))
    layout = dict(title = 'Top Most Associated Movies',
                  xaxis = dict(title = 'Movie Titles'),
                  yaxis = dict(title = 'Lift Score'))
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
    
top_association(n=20, min_support=0.10, min_size=10)
top_association(n=20, min_support=0.05, min_size=5)
