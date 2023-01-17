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
tag=pd.read_csv('/kaggle/input/movielens-20m-dataset/tag.csv')

rating=pd.read_csv('/kaggle/input/movielens-20m-dataset/rating.csv')

movies=pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv')

genome_scores=pd.read_csv('/kaggle/input/movielens-20m-dataset/genome_scores.csv')

link=pd.read_csv('/kaggle/input/movielens-20m-dataset/link.csv')

genome_tag=pd.read_csv('/kaggle/input/movielens-20m-dataset/genome_tags.csv')
movies.head()
link.head()
rating.head()
rating.shape
rating['userId'].value_counts().shape ## unique users
x=rating['userId'].value_counts()>500
y = x[x].index
y.shape
rating=rating[rating['userId'].isin(y)]
rating.shape
movie_details=movies.merge(rating,on='movieId')
movie_details.head()
movie_details.shape
movie_details.drop(columns=['timestamp'],inplace=True)
movie_details.shape
movie_details.head()
number_rating = movie_details.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns={'rating':'number of rating'},inplace=True)
number_rating.head()
df=movie_details.merge(number_rating,on='title')
df.shape
df.head()
df=df[df['number of rating']>=50] #selecting valuable books by ratings
df.drop_duplicates(['title','userId'],inplace=True)
df.shape
df.head()
df.drop(columns=['number of rating'],inplace=True)
df.head()
df['rating']=df['rating'].astype(int)
df.head()
movie_pivot=df.pivot_table(columns='userId',index='title',values='rating')
movie_pivot.shape
movie_pivot.fillna(0,inplace=True)
movie_pivot
from scipy.sparse import csr_matrix

movie_sparse=csr_matrix(movie_pivot)




from sklearn.neighbors import NearestNeighbors

model=NearestNeighbors( n_neighbors=7,algorithm='brute',metric='cosine')
model.fit(movie_sparse)
df.drop(columns=['genres','userId','rating'],inplace=True)
df.drop_duplicates(inplace=True)
df.to_csv('codf.csv',index=False)
distances,suggestions=model.kneighbors(movie_pivot.iloc[540,:].values.reshape(1,-1))
distances
suggestions
df1=df.copy()

ti=[]

for i in df1['title']:

    ti.append(i.split(' (')[0])

df1['title']=ti




for i in range(len(suggestions)):

    print(movie_pivot.index[suggestions[i]])

def reco(movie_name):

    movie_id=df1[df1['title']=='Toy Story'].drop_duplicates('title')['movieId'].values[0]

    distances,suggestions=model.kneighbors(movie_pivot.iloc[movie_id,:].values.reshape(1,-1))

    

    

    

    for i in range(len(suggestions)):

        return (movie_pivot.index[suggestions[i]])



res=reco("It Conquered the World")
for i in res:

    print(i)