# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv('../input/ratings.csv')
credits = pd.read_csv('../input/credits.csv')
keywords = pd.read_csv('../input/keywords.csv')
links = pd.read_csv('../input/links.csv')
movies.head()
genre_frequencies = [y for x in movies.genres for y in x]
from collections import Counter
genre_count = dict(Counter(genre_frequencies))
plt.figure(figsize=(30,15))
sns.barplot(x=list(genre_count.keys()),y=list(genre_count.values()))
plt.title("Genre Type Histogram", fontsize=30)
plt.xticks(fontsize=15)
plt.xlabel("Genre Types",fontsize=25)
plt.yticks(fontsize=15)
plt.figure(figsize=(30,15))
sns.distplot(movies.runtime[movies.runtime<420],color='c')
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Running Time',fontsize=25)
plt.ylabel('Frequencies',fontsize=25)
plt.title("Running Time histogram", fontsize=40)
plt.figure(figsize=(30,15))
sns.distplot(movies.runtime[movies.runtime<420],color='orange',bins=100)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Votes counts',fontsize=25)
plt.ylabel('Frequencies',fontsize=25)
plt.title("Vote counts Histogram", fontsize=40)
plt.figure(figsize=(30,15))
sns.distplot(list(movies.runtime[movies.runtime<420]),color='c',bins=100)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('Votes Average Score',fontsize=25)
plt.ylabel('Frequencies',fontsize=25)
plt.title("Vote Average Score Histogram", fontsize=40)
ratings.userId.value_counts().head()
tfid = TfidfVectorizer(stop_words='english')
movies.overview = movies.overview.fillna('')

### since making cos_sim matrix taking too long, temporaily used only 10000 rows
tfid_matrix = tfid.fit_transform(movies.overview.iloc[1:10000])
tfid_matrix.shape
cos_sim = cosine_similarity(tfid_matrix,tfid_matrix)
pd.DataFrame(cos_sim).head(10)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
indices.head()
idx = indices['Jumanji']
sim_movies = sorted(list(enumerate(cos_sim[idx])), key= lambda x: x[1], reverse=True)
sim_movies = sim_movies[1:11]
sim_num = [x[0] for x in sim_movies]
sim_value = [x[1] for x in sim_movies]
result = indices.iloc[sim_num]
### making recommend engine based on cosine similarities

def recommend_engine(title,cos_sim = cos_sim):
    idx = indices[title]
    sim_movies = sorted(list(enumerate(cos_sim[idx])), key= lambda x: x[1], reverse=True)
    sim_movies = sim_movies[1:11]
    sim_num = [x[0] for x in sim_movies]
    sim_value = [x[1] for x in sim_movies]
    result = indices.iloc[sim_num]
    result[0:10] = sim_value
    return(result)

### cosine sim value are represented with movie name
#### what is the recommended movies from Jumanji?
#### Most similar movies in terms of cosine similarities
recommend_engine('Jumanji')
movies.columns
credits.columns
### merge movies, keywords, credits data into movies sole dataset

movies = movies.drop([19730, 29503, 35587])

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
movies['id'] = movies['id'].astype('int')

movies = movies.merge(keywords, on='id')
movies = movies.merge(credits, on='id')
### strinfied features are splited into list

features = ['genres','keywords','cast','crew']

for feature in features:
    movies[feature] = movies[feature].apply(literal_eval)
### who is director? finding director function

def get_director(data):
    for x in data:
        if x['job'] == 'Director':
            return x['name']
    return np.nan
### making director columns

movies['director'] = movies.crew.apply(get_director)
### making get_list function
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names

    return []
movies.cast = movies.cast.apply(get_list)
movies.genres = movies.genres.apply(get_list)
movies.keywords = movies.keywords.apply(get_list)
### delete space within strings and change into lowercase 
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
### since our movies dataset is consiste of 3 columns of list and 1 string(director) column, 
### we divide function into two set.
features = ['cast','keywords','director','genres']
for feature in features:
    movies[feature] = movies[feature].apply(clean_data)
movies[features].head(10)
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' '
movies['soup'] = movies.apply(create_soup, axis=1)
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies['soup'])
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title'])
tmp = recommend_engine('The Godfather',cosine_sim2).index
list(tmp)
def rank_plot(movie_name, cos_sim=cos_sim):
    tmp = recommend_engine(movie_name,cos_sim)
    tmp2 = recommend_engine(movie_name,cosine_sim2)
    plt.figure(figsize=(10,5))
    sns.barplot(x = tmp[0:10], y=tmp.index)
    plt.title("Recommended Movies from  " + str.upper(movie_name) + " using cosine_sim", fontdict= {'fontsize' :20})
    plt.xlabel("Cosine Similarities")
    plt.show()
      
    plt.figure(figsize=(10,5))
    sns.barplot(x = tmp2[0:10], y=tmp2.index)
    plt.title("Recommended Movies from  " + str.upper(movie_name) + " using cosine_sim2", fontdict= {'fontsize' :20})
    plt.xlabel("Cosine Similarities")
    
    plt.show()
rank_plot("The Godfather")

rank_plot("Jumanji")
### importing rating dataset

rating = pd.read_csv("../input/ratings_small.csv")
### Checking data head

rating.head()
### Checking Data shape

rating.shape
### prepare Df which will record scores of movies

df = pd.DataFrame( index = rating.userId.unique() )
### Making df recording score of each user's record into df

for i in range(0,20000):
    ID = rating.loc[i,:].userId
    movieID = rating.loc[i,:].movieId
    movieScore = rating.loc[i,:].rating

    if movieID in list(df.columns):
        df.loc[ID, movieID] = movieScore
    else:
        df[movieID] = 0
        df.loc[ID,movieID] = movieScore
### shape of df (number of rows: number of users, number of columns : number of movies )

df.shape
### Checking data head

df.head()
### making cosine similarity matrix between users to users (671 by 671 in this case)

Filtering_cosim = cosine_similarity(df,df)
most_sim_user = sorted(list(enumerate(Filtering_cosim[100])), key=lambda x:x[1], reverse=True)[1]
most_sim_users = sorted(list(enumerate(Filtering_cosim[8])), key=lambda x: x[1], reverse=True)
most_sim_users = most_sim_users[1:11]
sim_users = [x[0] for x in most_sim_users]
print(sim_users)
candidates_movies = df.loc[sim_users,:]
def UBCF(user_num):
    ### finding most similar users among matrix

    most_sim_users = sorted(list(enumerate(Filtering_cosim[user_num])), key=lambda x: x[1], reverse=True)
    most_sim_users = most_sim_users[1:11]

    ### user index and their similairity values 

    sim_users = [x[0] for x in most_sim_users]
    sim_values = [x[1] for x in most_sim_users]

    ### among users having most similar preferences, finding movies having highest average score
    ### however except the movie that original user didn't see

    candidates_movies = df.loc[sim_users,:]

    candidates_movies.mean(axis=0).head()

    mean_score = pd.Series(candidates_movies.mean(axis=0))
    mean_score = mean_score.sort_values(axis=0, ascending=False)
    
    recom_mov = list(mean_score.iloc[0:10].keys())
    for i in recom_mov:
        recom_mov_title = movies.loc[movies.id.isin(recom_mov),:].title
        recom_mov_title
    return(recom_mov_title)
UBCF(400)
UBCF(1)