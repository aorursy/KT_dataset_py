# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ast import literal_eval

movies=pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')

movies.head()

movies.isnull().sum()

movies.dtypes

movies.columns 

movies=movies.rename(columns={'id':'movieId'})

movies['movieId']=pd.to_numeric(movies['movieId'],errors='coerce')

movies=movies[['movieId','title','original_language']]

movies['original_language'].unique()

ratings=pd.read_csv('/kaggle/input/the-movies-dataset/ratings_small.csv')

ratings.head()

ratings.dtypes

ratings.isnull().sum()

ratings=ratings.drop(['timestamp'],axis=1)

duplicate_movies = movies.groupby('title').filter(lambda x: len(x) == 2)

duplic_ids = duplicate_movies['movieId'].values

#Duplicated titles

duplicate_movies = duplicate_movies[['movieId','title']]

review_count=pd.DataFrame(ratings[ratings['movieId'].isin(duplic_ids)]['movieId'].value_counts())

review_count.reset_index()

duplicate_df=pd.merge(duplicate_movies,ratings,on='movieId',how='left')

duplicate_df

duplic_ids=duplicate_df.drop_duplicates(subset='title',keep='last',inplace=False)['movieId']

duplic_ids

movies=movies[~movies['movieId'].isin(duplic_ids)]

movies

ratings=ratings[~ratings['movieId'].isin(duplic_ids)]

ratings

ratings.shape

#ratings=ratings.iloc[1:100000]

ratings[ratings['movieId']==1]

movies['movieId'].unique()



sel_movie=movies[movies['movieId']==480].loc[:,['title']]

sel_movie

movies['movieId'].unique()

movie=pd.merge(movies,ratings,on='movieId',how='left')

movie.head(50)

movie.shape

movie=movie[1:20000]

cross=pd.crosstab(movie['userId'],movie['title'])

cross.head(10)

from sklearn.decomposition import PCA

pca=PCA(n_components=4)

pca_data=pca.fit_transform(cross)

ps=pd.DataFrame(pca_data)

ps

tocluster=pd.DataFrame(ps[[0,1,2]])

tocluster

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['figure.figsize'] = (16, 9)

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(tocluster[0], tocluster[2], tocluster[1])

plt.title('Data points in 3D PCA axis', fontsize=20)

plt.show()

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

scores=[]

inertia_list=np.empty(8)

for i in range(2,8):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(tocluster)

    inertia_list[i] = kmeans.inertia_

    scores.append(silhouette_score(tocluster, kmeans.labels_))

plt.plot(range(0,8),inertia_list,'-o')

plt.xlabel('Number of cluster')

plt.axvline(x=5, color='blue', linestyle='--')

plt.ylabel('Inertia')

plt.show()   

clusterer = KMeans(n_clusters=4,random_state=30).fit(tocluster)

centers = clusterer.cluster_centers_

c_preds = clusterer.predict(tocluster)

centers

c_preds

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(tocluster[0], tocluster[2], tocluster[1], c = c_preds)

plt.title('Data points in 3D PCA axis', fontsize=20)

plt.show()

fig = plt.figure(figsize=(10,8))

plt.scatter(tocluster[1],tocluster[0],c = c_preds)

for ci,c in enumerate(centers):

    plt.plot(c[1], c[0], 'o', markersize=8, color='red', alpha=1)

plt.xlabel('x_values')

plt.ylabel('y_values')

plt.title('Data points in 2D PCA axis', fontsize=20)

plt.show()

cross['cluster']=c_preds

cross.head(10)

c0 = cross[cross['cluster']==0].drop('cluster',axis=1).mean()

c1 = cross[cross['cluster']==1].drop('cluster',axis=1).mean()

c2 = cross[cross['cluster']==2].drop('cluster',axis=1).mean()

c3 = cross[cross['cluster']==3].drop('cluster',axis=1).mean()

c0.sort_values(ascending=False)[0:15]

c1.sort_values(ascending=False)[0:15]

c2.sort_values(ascending=False)[0:15]

c3.sort_values(ascending=False)[0:15]

#movie=movie.iloc[1:100000]

f=['count','mean']

average=movie.groupby('movieId')['rating'].agg(f)

average=average.sort_values(by='count',ascending=False)

average.head(20)

count1=ratings['userId'].value_counts()

count1.quantile(0.5)

#ratings=ratings[ratings['userId'].isin(count1[count1>=30].index)]

counts=ratings['rating'].value_counts()

counts.quantile(0.65)

#ratings=ratings[ratings['rating'].isin(counts[counts>=300000.0].index)]

df_pivot=pd.pivot_table(ratings,values='rating',index='userId',columns='movieId')



df_pivot.shape

df_pivot.columns

#Pearson R Recommendation

rating_p=df_pivot[480]

rating_p

similar_to_rating_p=df_pivot.corrwith(rating_p)

similar_to_rating_p_df=pd.DataFrame(similar_to_rating_p,columns=['PearsonR'])

similar_to_rating_p_df

similar_to_rating_p_df.dropna(inplace=True)

similar_to_rating_p_df

corr_summary=similar_to_rating_p_df.join(average['count'])

corr_summary=corr_summary[(corr_summary['PearsonR']>0.5) & (corr_summary['PearsonR']<=0.99)]

corr_summary.sort_values('PearsonR',ascending=False).head(20)

movie_corr=pd.DataFrame([4628,46967,5833,166,1858],np.arange(5),columns=['movieId'])

corr_movie=pd.merge(movie_corr,movies,on='movieId')

print("recommendation for{}",sel_movie)

sel_movie

corr_movie

#Kmean Recommendation

movie_drop=movie.dropna(axis=0,subset=['title'])

movie_drop_df=movie_drop[['title','rating']]

movie_combined=movie_drop_df.groupby('title').count()

movie_combined.head(50)

total_movie_combined=pd.merge(movie_drop,movie_combined,on='title')

total_movie_combined.head(100)

movie_combined['rating'].quantile(np.arange(.9,1,.01))

popularity_threshold=30

rating_popular=total_movie_combined.drop(['rating_x'],axis=1)

rating_popular=rating_popular.drop_duplicates(['userId'])

rating_popular.head(50)

rating_p=pd.pivot(rating_popular,index='title',values='rating_y',columns='userId')

rating_p=rating_p.fillna(0)

rating_p

from scipy.sparse import csr_matrix

user_matrix=csr_matrix(rating_p.values)

user_matrix

from sklearn.neighbors import NearestNeighbors

model=NearestNeighbors(metric='cosine',algorithm='brute')

model.fit(user_matrix)

query_index = np.random.choice(rating_p.shape[0])

rating_p.iloc[query_index,]

distances, indices = model.kneighbors(rating_p.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 5)

for i in range(0, len(distances.flatten())):

    if i == 0:

        print('Recommendations for {0}:\n'.format(rating_p.index[query_index]))

    else:

        print('{0}: {1}, with distance of {2}:'.format(i, rating_p.index[indices.flatten()[i]], distances.flatten()[i]))

#Recommending with film overviews and taagline

mv=pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')

mv=mv.iloc[1:20000]

#mv=mv.rename(columns={'id':'movieId'})

mv['id']=pd.to_numeric(mv['id'],errors='coerce')

mv.head()

smd=mv.copy()

smd['tagline']=smd['tagline'].fillna('')

smd['description']=smd['tagline']+smd['overview']

smd['description']=smd['description'].fillna('')

smd['description']

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tf=TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

tfidf = tf.fit_transform(smd['description'])

tfidf.shape

tfidf

#Cosine Similarity

from sklearn.metrics.pairwise import linear_kernel

cosine=linear_kernel(tfidf,tfidf)

cosine

#Function to return top movies based on cosine simalirity score

smd=smd.reset_index()

title=smd['title']

title

title_index=pd.Series(smd.index,index=smd['title'])

idx = title_index['Miss Bala']

sim_scores = list(enumerate(cosine[idx]))

sim_scores

sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

sim_scores = sim_scores[1:31]

movie_indices = [i[0] for i in sim_scores]

movie_indices    

def recommendation(titles):

    idx = title_index[titles]

    sim_scores = list(enumerate(cosine[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    movie_indices = [i[0] for i in sim_scores]

    return title.iloc[movie_indices]

recommendation('The Godfather').head(10)

#Recommending with cast/genres and keywords

movies=pd.read_csv('/kaggle/input/the-movies-dataset/movies_metadata.csv')

movies.head()

credits=pd.read_csv('/kaggle/input/the-movies-dataset/credits.csv')

movies['id']=pd.to_numeric(movies['id'],errors='coerce')

movies=pd.merge(movies,credits,on='id')

movies

keywords=pd.read_csv('/kaggle/input/the-movies-dataset/keywords.csv')

keywords.head()

movies=pd.merge(movies,keywords,on='id')

movies.head()

movies=movies.iloc[1:20000]

movies['cast']=movies['cast'].apply(literal_eval)

movies['cast']

movies['crew']=movies['crew'].apply(literal_eval)

movies['crew']

movies['keywords']=movies['keywords'].apply(literal_eval)

movies['keywords']

movies['genres']=movies['genres'].apply(literal_eval)



def get_directors(x):

    for i in x:

        if i['job']=='Director':

            return i['name']

        return np.nan

movies['crew']=movies['crew'].apply(get_directors)

movies['crew']=movies['crew'].astype(str)

def get_list(x):

    if isinstance(x,list):

        names=[i['name'] for i in x]

        if len(names)>3:

            names=names[:3]

        return names

    return []

features=['cast','keywords','genres']

for i in features:

    movies[i]=movies[i].apply(get_list)

movie_df=movies[['title','cast','crew','keywords','genres']]  

def clean_data(x):

    if isinstance(x,list):

        for i in x:

            return[str.lower(i.replace(' ',''))]

    if isinstance(x,str):

        return str.lower(x.replace(' ',''))

    else:

        return ''

for i in features:

    movie_df[i]=movie_df[i].apply(clean_data)

def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['crew'] + ' ' + ' '.join(x['genres'])

movie_df['soup'] = movie_df.apply(create_soup, axis=1)

movie_df['soup']

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()

cm=cv.fit_transform(movie_df['soup'])

cm

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(cm,cm)

cosine_sim

title=movie_df['title']

movie_df=movie_df.reset_index()

title_index=pd.Series(movie_df.index,index=movie_df['title'])

def recommendation(titles):

    idx = title_index[titles]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    movie_indices = [i[0] for i in sim_scores]

    return title.iloc[movie_indices]

recommendation('The Dark Knight').head(10)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

os.listdir('../input')

# Any results you write to the current directory are saved as output.