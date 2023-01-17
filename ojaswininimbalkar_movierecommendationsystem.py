import pandas as pd

import numpy as np
movies=pd.read_csv("/kaggle/input/newdataset/movies.csv")

ratings=pd.read_csv("/kaggle/input/newdataset/ratings.csv")
movies.head()
ratings.head()
df=pd.merge(ratings,movies, on='movieId')
df.head(10)
df.shape
Ratings_avg=pd.DataFrame(df.groupby('title')['rating'].mean())

Ratings_avg.head()

#to print count of ratings





Ratings_avg['Rating count']=pd.DataFrame(df.groupby('title')['rating'].count())

Ratings_avg.head()
#Most rated movies of all time:



df.groupby('title')['rating'].count().sort_values(ascending=False).head()
userRatings=df.pivot_table(index='userId',columns='title',values='rating')

userRatings.head()
#Input user's Favourite movie to show the correlation calculation

movieName=input()
#Correlation Calculator



correlations=userRatings.corrwith(userRatings[movieName])

correlations

recommendation = pd.DataFrame(correlations,columns=['Correlation'])

recommendation.dropna(inplace=True)

recommendation = recommendation.join(Ratings_avg['Rating count'])

recommendation.head()
rec = recommendation[recommendation['Rating count']>100].sort_values('Correlation',ascending=False).reset_index()
rec.head()
rec = rec.merge(movies,on='title', how='left')

rec.head()