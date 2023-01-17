import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans

import json
path1='../input/tmdb_5000_movies.csv'

dl1=pd.read_csv(path1)

dl1.head()
path2='../input/tmdb_5000_credits.csv'

dl2=pd.read_csv(path2)

dl2.head()
dl1.columns
dl2.columns
dl2.columns=['id','title','cast','crew']
dl1= dl1.merge(dl2,on='id')
dl1.shape
dl1.head()
dl1.dtypes
dl1.isnull().sum()
dl1.nunique()
plt.rcParams['figure.figsize']=(20,20)

hm=sns.heatmap(dl1[['budget', 'popularity', 'revenue','runtime', 'vote_average', 'vote_count']].corr(), annot = True)
dl1['rentability']=dl1['revenue']-dl1['budget']
dl1.describe()
n = 0 

for x in ['rentability','vote_average','vote_count']:

    for y in ['rentability','vote_average','vote_count']:

        n += 1

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

        sns.regplot(x = x , y = y , data = dl1)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )

plt.show()
n = 0 

for cols in ['rentability','vote_average','vote_count']:

    n += 1 

    plt.subplot(1 , 3 , n)

    sns.violinplot(x = cols  , data = dl1 )

plt.show()
plt.scatter(x = 'rentability' , y = 'vote_average' , data = dl1)

plt.xlabel('rent'), plt.ylabel('vote') 

plt.legend()

plt.show()
plt.scatter(x = 'rentability' , y = 'vote_count' , data = dl1)

plt.xlabel('rent'), plt.ylabel('vote') 

plt.legend()

plt.show()
plt.scatter(x = 'vote_average' , y = 'vote_count' , data = dl1)

plt.xlabel('rent'), plt.ylabel('vote') 

plt.legend()

plt.show()
max_a = dl1['vote_average'].max()

min_a = dl1['vote_average'].min()



max_c = dl1['vote_count'].max()

min_c = dl1['vote_count'].min()



dl1['score'] = 2.5*(dl1['vote_average'] - min_a) / (max_a - min_a)+2.5*(dl1['vote_count'] - min_c) / (max_c - min_c)
dl1.head()
dl1 = dl1.sort_values('score', ascending=False)
dl1[['title_x','score']].head()
dl1['score'].mean()

sns.violinplot(x = 'score'  , data = dl1 )

plt.show()
dl1['overview'].head()
from sklearn.feature_extraction.text import TfidfVectorizer
TV = TfidfVectorizer(stop_words='english')

dl1['overview'] = dl1['overview'].fillna('')

TV_matrix = TV.fit_transform(dl1['overview'])
TV_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
indice = pd.Series(dl1.index, index=dl1['title_x']).drop_duplicates()

sim = linear_kernel(TV_matrix, TV_matrix)



def recommandation(title):    

    idx = indice[title]

    similarity = list(enumerate(sim[idx]))

    similarity = similarity[0:10]

    movie_indices = [i[0] for i in similarity]

    return dl1[['title_x','score']].iloc[movie_indices].drop(idx)
recommandation('Interstellar')