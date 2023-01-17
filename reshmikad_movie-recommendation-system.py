# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from surprise import Reader, Dataset, SVD
from sklearn.metrics.pairwise import cosine_similarity
from surprise.model_selection import cross_validate
from ast import literal_eval
df=pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')
df.head()
C= df['vote_average'].mean()
print('Average Vote: ', C)
m= df['vote_count'].quantile(0.95)
print('Vote Count above 95% : ', m)
qualified = df.copy().loc[df['vote_count'] >= m]
qualified.shape
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

qualified['score'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('score', ascending=False).head(100)
qualified[['title', 'vote_count', 'vote_average', 'score']].head(10)
df['genres']
def genre_based(genre, percentile=0.85):
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) ] [['title', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['score'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average'])  + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('score', ascending=False).head(250)
    
    return qualified
genre_based('Romance').head(15)
df_small =pd.read_csv('/kaggle/input/the-movies-dataset/links_small.csv')
df_small
df_small = df_small[df_small['tmdbId'].notnull()]['tmdbId'].astype('int')
df = df.drop([19730, 29503, 35587])
df['id'] = df['id'].astype('int')
sdf = df[df['id'].isin(df_small)]
sdf.shape
sdf
sdf['description'] = sdf['overview'] + sdf['tagline']
sdf['description'] = sdf['description'].fillna('')
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(sdf['description'])
tfidf_matrix.shape
cos_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
sdf = sdf.reset_index()
indices = pd.Series(sdf.index, index=sdf['title']).drop_duplicates()

def get_recommendations(title, cos_sim=cos_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return sdf['title'].iloc[movie_indices]
get_recommendations('The Baby-Sitters Club')
get_recommendations('The Godfather')
credits = pd.read_csv('/kaggle/input/the-movies-dataset/credits.csv')
keywords = pd.read_csv('/kaggle/input/the-movies-dataset/keywords.csv')
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
df['id'] = df['id'].astype('int')

df.shape
df = df.merge(credits, on='id')
df = df.merge(keywords, on='id')
sdf = df[df['id'].isin(df_small)]
sdf.shape
features = ['cast', 'crew', 'keywords']
for feature in features:
    sdf[feature] = sdf[feature].apply(literal_eval)

sdf['cast_size'] = sdf['cast'].apply(lambda x: len(x))
sdf['crew_size'] = sdf['crew'].apply(lambda x: len(x))
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

sdf['director'] = sdf['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    sdf[feature] = sdf[feature].apply(get_list)
sdf[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
s = sdf.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s[:10]
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else: 
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    sdf[feature] = sdf[feature].apply(clean_data)
    
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
sdf['soup'] = sdf.apply(create_soup, axis=1)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(sdf['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

sdf = sdf.reset_index()
indices = pd.Series(sdf.index, index=sdf['title'])
get_recommendations('The Prestige', cosine_sim2)
reader = Reader()
ratings = pd.read_csv('/kaggle/input/the-movies-dataset/ratings_small.csv')
ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
print(data.df.head())
#.split(n_folds=5)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()
svd.fit(trainset)

ratings[ratings['userId'] == 1]
svd.predict(1, 302, 3)
def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
    
id_map = pd.read_csv('/kaggle/input/the-movies-dataset/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(sdf[['title', 'id']], on='id').set_index('title')

indices_map = id_map.set_index('id')

def hpred(userId, title):
    index = indices[title]
    Id = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']
    
    recc = list(enumerate(cos_sim[int(index)]))
    recc = sorted(recc, key=lambda x: x[1], reverse=True)
    recc = recc[1:25]
    movie_indices = [i[0] for i in recc]
    movies = sdf.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)
hpred(1, 'Bride Wars')
hpred(500, 'Bride Wars')