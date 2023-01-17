import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

import gc

from datetime import datetime 

from sklearn.model_selection import train_test_split

import os

pd.set_option('display.max_columns', 100)
PATH="../input"

print(os.listdir(PATH))
ratings_df = pd.read_csv(os.path.join(PATH,"ratings_small.csv"), low_memory=False)
links_df = pd.read_csv(os.path.join(PATH,"links_small.csv"), low_memory=False)
movies_metadata_df = pd.read_csv(os.path.join(PATH,"movies_metadata.csv"), low_memory=False)
credits_df = pd.read_csv(os.path.join(PATH,"credits.csv"), low_memory=False)
keywords_df = pd.read_csv(os.path.join(PATH,"keywords.csv"), low_memory=False)
print("Ratings data contains {} rows and {} columns".format(ratings_df.shape[0], ratings_df.shape[1]))

print("Links data contains {} rows and {} columns".format(links_df.shape[0], links_df.shape[1]))

print("Movie metadata contains {} rows and {} columns".format(movies_metadata_df.shape[0], movies_metadata_df.shape[1]))

print("Credits data contains {} rows and {} columns".format(credits_df.shape[0], credits_df.shape[1]))

print("Keywords data contains {} rows and {} columns".format(keywords_df.shape[0], keywords_df.shape[1]))
ratings_df.head()
links_df.head()
movies_metadata_df.head()
keywords_df.head()
from ast import literal_eval

# Returns the list top l elements or entire list; whichever is more.

def get_list(x, l=5):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        #Check if more than l elements exist. If yes, return only first three. If no, return entire list.

        if len(names) > l:

            names = names[:l]

        return names



    #Return empty list in case of missing/malformed data

    return []



movies_metadata_df['genres'] = movies_metadata_df['genres'].apply(literal_eval)

movies_metadata_df['genres'] = movies_metadata_df['genres'].apply(get_list)
pd.DataFrame({'feature':ratings_df.dtypes.index, 'dtype':ratings_df.dtypes.values})
movies_metadata_df.head()
pd.DataFrame({'feature':movies_metadata_df.dtypes.index, 'dtype':movies_metadata_df.dtypes.values})
ratings_df.describe()
import datetime

min_time = datetime.datetime.fromtimestamp(min(ratings_df.timestamp)).isoformat()

max_time = datetime.datetime.fromtimestamp(max(ratings_df.timestamp)).isoformat()

print('Timestamp for ratings from {} to {}:'.format(min_time, max_time))
def check_missing(data_df):

    total = data_df.isnull().sum().sort_values(ascending = False)

    percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()



check_missing(ratings_df)
check_missing(movies_metadata_df)
movies_metadata_df.dropna(subset=['title'], inplace=True)

check_missing(movies_metadata_df)
movies_metadata_df['id'] = pd.to_numeric(movies_metadata_df['id'])
ratings_df.shape
ratings_df = ratings_df.merge(movies_metadata_df[['id']], left_on=['movieId'], right_on=['id'], how='inner')
ratings_df.shape
ratings_df['time_dt'] = ratings_df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
ratings_df.head()
ratings_df['year'] = ratings_df['time_dt'].dt.year

ratings_df['month'] = ratings_df['time_dt'].dt.month

ratings_df['day'] = ratings_df['time_dt'].dt.day

ratings_df['dayofweek'] = ratings_df['time_dt'].dt.dayofweek
ratings_df[['year', 'month', 'day', 'dayofweek']].describe()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(18,3))

s = sns.boxplot(ax = ax1, y="year", data=ratings_df, palette="Greens",showfliers=True)

s = sns.boxplot(ax = ax2, y="month", data=ratings_df, palette="Blues",showfliers=True)

s = sns.boxplot(ax = ax3, y="day", data=ratings_df, palette="Reds",showfliers=True)

s = sns.boxplot(ax = ax4, y="dayofweek", data=ratings_df, palette="Reds",showfliers=True)

plt.show()
dt = ratings_df.groupby(['year'])['rating'].count().reset_index()

fig, (ax) = plt.subplots(ncols=1, figsize=(12,6))

plt.plot(dt['year'],dt['rating']); plt.xlabel('Year'); plt.ylabel('Number of votes'); plt.title('Number of votes per year')

plt.show()
dt = ratings_df.groupby(['year'])['rating'].mean().reset_index()

fig, (ax) = plt.subplots(ncols=1, figsize=(12,6))

plt.plot(dt['year'],dt['rating']); plt.xlabel('Year'); plt.ylabel('Average ratings'); plt.title('Average ratings per year')

plt.show()
fig, (ax) = plt.subplots(ncols=1, figsize=(12,4))

s = sns.boxplot(x='year', y="rating", data=ratings_df, palette="Greens",showfliers=True)

plt.show()
fig, (ax) = plt.subplots(ncols=1, figsize=(10,4))

s = sns.boxplot(x='month', y="rating", data=ratings_df, palette="Blues",showfliers=True)

plt.show()
fig, (ax) = plt.subplots(ncols=1, figsize=(6,4))

s = sns.boxplot(x='dayofweek', y="rating", data=ratings_df, palette="Reds",showfliers=True)

plt.show()
print("There is a total of {} users, with an average number of {} votes.".format(ratings_df.userId.nunique(), \

                                                round(ratings_df.shape[0]/ratings_df.userId.nunique()),2))
print("Top 5 voting users:\n")

tmp = ratings_df.userId.value_counts()[:5]

pd.DataFrame({'Votes':tmp.values, 'Id':tmp.index})
tmp = ratings_df.userId.value_counts()

df = pd.DataFrame({'Votes':tmp.values, 'Id':tmp.index})

print("There are {} users that voted only once.".format(df[df['Votes']==1].nunique().values[0]))
tmp = ratings_df.groupby(['userId'])['rating'].mean().reset_index()

tmp['rating'] = tmp['rating'].apply(lambda x: round(x,3))

df_max = tmp[tmp['rating']==5]

df_min = tmp[tmp['rating']==0.5]

print("Users giving only '5': {}\nUsers giving only '0.5':{}".format(df_max.shape[0], df_min.shape[0]))
mean_rating = round(ratings_df['rating'].mean(),3)

print("Average value of rating is {}.".format(mean_rating))

print("There are {} users that have their average score with the overall average score (approx. with 3 decimals).".format(\

                            tmp[tmp['rating']==mean_rating]['userId'].nunique()))
print("There is a total of {} movies, with an average number of {} votes.".format(ratings_df.movieId.nunique(), \

                                                round(ratings_df.shape[0]/ratings_df.movieId.nunique()),2))
print("Top 10 voted movies:\n")

tmp = ratings_df.movieId.value_counts()[:10]

pd.DataFrame({'Votes':tmp.values, 'id':tmp.index})
top_10 = pd.DataFrame({'Votes':tmp.values, 'id':tmp.index}).merge(movies_metadata_df)

top_10
tmp = ratings_df.movieId.value_counts()

df = pd.DataFrame({'Votes':tmp.values, 'Id':tmp.index})

print("There are {} movies that were voted only once.".format(df[df['Votes']==1].nunique().values[0]))
tmp = ratings_df.groupby(['movieId'])['rating'].mean().reset_index()

tmp['rating'] = tmp['rating'].apply(lambda x: round(x,3))

df_max = tmp[tmp['rating']==5]

df_min = tmp[tmp['rating']==0.5]

print("Movies with only '5': {}\nMovies with only '0.5':{}".format(df_max.shape[0], df_min.shape[0]))
mean_rating = round(ratings_df['rating'].mean(),3)

print("Average value of rating is {}.".format(mean_rating))

print("There are {} movies that have their average score with the overall average score (approx. with 3 decimals).".format(\

                            tmp[tmp['rating']==mean_rating]['movieId'].nunique()))
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=17,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()



show_wordcloud(movies_metadata_df['genres'], title = 'Movie Genres Prevalence in The Movie Dataset')
tmp = ratings_df.groupby(['movieId'])['rating'].mean()

R = pd.DataFrame({'id':tmp.index, 'R': tmp.values})

tmp = ratings_df.groupby(['movieId'])['rating'].count()

v = pd.DataFrame({'id':tmp.index, 'v': tmp.values})

C = ratings_df['rating'].mean()
m_df = movies_metadata_df.merge(R, on=['id'])

m_df = m_df.merge(v, on=['id'])

m_df['C'] = C

m= m_df['v'].quantile(0.9)

m_df['m'] = m
m_df.head()
m_df['IMDB'] = (m_df['v'] / (m_df['v'] + m_df['m'])) * m_df['R'] + (m_df['m'] / (m_df['v'] + m_df['m'])) * m_df['C']
m_df.sort_values(by=['IMDB'], ascending=False).head(10)
m_df[['title', 'IMDB']].sort_values(by=['IMDB'], ascending=False).head(10)
m_df['R_x_v'] = m_df['R'] * m_df['v']
m_df[['title', 'v']].sort_values(by=['v'], ascending=False).head(10)
m_df[['title', 'R_x_v']].sort_values(by=['R_x_v'], ascending=False).head(10)
del tmp, top_10

gc.collect()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english',max_features=10000)

tokens = m_df[['title']]

tokens['title'] = tokens['title'].fillna('')

tfidf_matrix = tfidf.fit_transform(tokens['title'])

print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print(cosine_sim.shape)

indices = pd.Series(tokens.index, index=tokens['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):

    # index of the movie that matches the title

    idx = indices[title]



    # similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # movie indices

    movie_indices = [i[0] for i in sim_scores]



    # top 10 most similar movies

    return tokens['title'].iloc[movie_indices]

get_recommendations('The Million Dollar Hotel')
get_recommendations('Sleepless in Seattle')
tfidf = TfidfVectorizer(stop_words='english',max_features=10000)

tokens = m_df[['title']]

tokens['title'] = tokens['title'].fillna('')

tfidf_matrix = tfidf.fit_transform(tokens['title'])

print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print(cosine_sim.shape)

indices = pd.Series(tokens.index, index=tokens['title']).drop_duplicates()
def get_imdb_score(df, indices):

    # select the data from similarity indices

    tmp = df[df.id.isin(indices)]

    # sort the data by IMDB score

    tmp = tmp.sort_values(by='IMDB', ascending=False)

    # return title and IMDB score

    return tmp[['title','IMDB']].head(10)
def get_10_recommendations_simpol(title, cosine_sim=cosine_sim):

    # index of the movie that matches the title

    idx = indices[title]



    # similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # scores of the 20 most similar movies

    sim_scores = sim_scores[1:21]

    

    # movie indices

    movie_indices = [i[0] for i in sim_scores]



    # get popularity scores

    pop_scores = get_imdb_score(m_df, movie_indices)

    

    return list(pop_scores['title'])

get_10_recommendations_simpol('The Million Dollar Hotel')
get_10_recommendations_simpol('Judgment Night')
get_10_recommendations_simpol('Fahrenheit 9/11')