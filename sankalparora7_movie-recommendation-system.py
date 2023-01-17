# This is a movie recommendation system 
import numpy as np
import pandas as pd
import warnings

#helps us to ignore any warnings if we get any 
warnings.filterwarnings('ignore')
"""here we will be using only two files that is u.data and u.info"""
column_name=["user_d","item_id","rating","timstamp"]
df=pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.data",sep='\t',names=column_name)
# here the file is tab seperated file.
df.head()
df.shape
# see hoe many unique users and movies are there
df['user_d'].nunique()
df['item_id'].nunique()
#npw to see which movie it is with the item-id, read the u.item file
movies_name=pd.read_csv("../input/movielens-100k-dataset/ml-100k/u.item",sep='\|',header=None)
movies_name
movies_name.shape
movies_name=movies_name[[0,1]]
movies_name.columns=['item_id','title']
movies_name.head()
# now merge the 2 dataframes
data=pd.merge(df,movies_name,on='item_id')
data
data.tail()
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style('white')
# find the avg rating for a movie
data.groupby('title').mean()
data.groupby('title').mean()['rating'].sort_values(ascending=False)
data.groupby('title').count()['rating'].sort_values(ascending=False)
ratings=pd.DataFrame(data.groupby('title').mean()['rating'])
ratings['No of ratings']=data.groupby('title').count()['rating']
ratings
ratings.sort_values(by='rating',ascending=False)
# we will have to remove the ratings with 5 star and have only 1 review
plt.figure(figsize=(10,7))
plt.hist(ratings['No of ratings'],bins=70)
plt.show()
plt.hist(ratings['rating'],bins=70)
plt.show()
sns.jointplot(x='rating' ,y='No of ratings' ,data=ratings,alpha=0.5)
#make a movie matrix
moviemat=data.pivot_table(index='user_d',columns='title',values='rating')
moviemat
#select any movie and check its user reviews
starwaruser_ratings=moviemat['Star Wars (1977)']
starwaruser_ratings
#now check the correlation of this movie with the movie matrix
similar_to_starwars=moviemat.corrwith(starwaruser_ratings)
similar_to_starwars
corr_starwars=pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars
#clean this data, like remove the nan values 
corr_starwars.dropna(inplace=True)
corr_starwars
corr_starwars.sort_values('Correlation',ascending=False).head(n=10)
#top 10 movies with highest corr
ratings
corr_starwars
#join the no of ratings column with the corr_Starwars
corr_starwars=corr_starwars.join(ratings['No of ratings'])
corr_starwars
corr_starwars[corr_starwars['No of ratings']>100].sort_values('Correlation',ascending=False)
#the same movies are seen on google too
def predict_movies(movie_name):
    movie_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_ratings)
    
    corr_movie=pd.DataFrame(similar_to_movie,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    
    corr_movie=corr_movie.join(ratings['No of ratings'])
    predictions=corr_movie[corr_movie['No of ratings']>100].sort_values('Correlation',ascending=False)
    return predictions
    
predictions=predict_movies("Titanic (1997)")
predictions.head()















