import numpy as np

import pandas as pd
columns_name=['user_id','item_id','rating','timestamp']

df=pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.data",sep="\t",names=columns_name)        

#u.data is a tsv file (tab separated values)

print(df.head())

df.shape
movies=pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.item",sep="\|",header= None)

print(movies.shape)

movies.head()
#There are 24 columns bit we just need to extract two columns from this dataset to get the movie names corresponding to each item_id

movies=movies[[0,1]]

movies.head()
#Give column names to the columns of this data frame also.

movies.columns=['item_id','title'] 

movies.head(1)
df=pd.merge(df,movies,on="item_id")
df.groupby("title").mean()['rating'].sort_values(ascending=False)
df.groupby("title").count()["rating"].sort_values(ascending=False)
ratings=pd.DataFrame(df.groupby("title").mean()['rating'])

ratings['number of ratings']=pd.DataFrame(df.groupby("title").count()["rating"])

print(ratings.head())
ratings.sort_values(by='rating', ascending=False)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("dark")
plt.figure(figsize=(12,8))

plt.hist(ratings['number of ratings'], bins=70,)

plt.show

plt.hist(ratings['rating'],bins=70)

plt.show
sns.jointplot(x='rating',y='number of ratings',data=ratings,alpha=0.5)
moviematrix=df.pivot_table(index="user_id",columns="title",values='rating')

print(moviematrix)
starwars_user_ratings=moviematrix['Star Wars (1977)']

starwars_user_ratings.head()
#finding correlation of starwars movie withh all other movies

similar_to_starwars=moviematrix.corrwith(starwars_user_ratings)    
similar_to_starwars

corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])

corr_starwars.head()
corr_starwars.dropna(inplace=True)
corr_starwars.sort_values('correlation',ascending=False).head(10)
corr_starwars=corr_starwars.join(ratings['number of ratings'])
corr_starwars.head()
corr_starwars[corr_starwars['number of ratings']>100].sort_values('correlation',ascending=False)
def predict_movies(movie_name):

    movie_user_ratings=moviematrix[movie_name]

    similar_to_movie=moviematrix.corrwith(movie_user_ratings)

    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])

    corr_movie.dropna(inplace=True)

    corr_movie=corr_movie.join(ratings['number of ratings'])

    predictions=corr_movie[corr_movie['number of ratings']>100].sort_values('correlation',ascending=False)

    return predictions

predictions=predict_movies("As Good As It Gets (1997)")  #any movie name from the data set can be put here

predictions.head()