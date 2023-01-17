#Input data

import pandas as pd

import numpy as np

names = ["movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western | "]

names = [i.split(' | ') for i in names][0]

movie_data = pd.read_csv("../input/ml-100k/u.item",delimiter="|",encoding="437",names=names)

movie_data['video release date'] = movie_data['release date']

rating_data =pd.read_csv("../input/ml-100k/u.data",delimiter="\t",names = ["user id","item id","rating" ,"timestamp"])
#Rename Columns , Drop Error Columns

rating_data = rating_data.rename(columns={'item id':'movie id'})

movie_data = movie_data.drop('',axis=1)
#Merge Dataframe

full_data = movie_data.merge(rating_data,on='movie id',how="inner")

full_data = full_data.drop_duplicates('movie id')
#Observe NaN

full_data[full_data.isnull().any(1)]
#drop record containing NaN

full_data = full_data.dropna(axis=0,subset=['release date']).reset_index().drop('index',axis=1)

full_data.head()
#Separate title with release year

full_data['release date'] = full_data['movie title'].str.split('(').str[1].str.replace(')','')

full_data['movie title'] = full_data['movie title'].str.split('(').str[0]
full_data = full_data.reset_index().drop('index',axis=1)
#calculate each genre excat score

genres_titles = np.array(full_data.columns[6:24])
genre_score = pd.DataFrame()

for gen in genres_titles:

    genre_score[gen] = pd.DataFrame(full_data[gen] * full_data['rating'],columns=[gen])

genre_score.head()
genre_score.describe()
#plot the describe more easy to see more detail

import matplotlib.pyplot as plt

%matplotlib inline
genre_score.describe().ix['mean'].plot(kind='bar',title='Mean of rating for genre')
#separate the high scoring feature films

good_dra_index = (full_data['Drama']==1)&(full_data['rating']>=4)

good_dra = full_data[good_dra_index]
good_dra.shape
#Calculate the number of high scores for each type of film, except for the drama 

genre_count = good_dra[genres_titles].apply(np.sum,axis=0)

genre_count = genre_count.drop('Drama')
genre_count.plot(kind="bar",title="high rating with drama")