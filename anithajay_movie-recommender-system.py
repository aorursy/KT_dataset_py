import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
col_names= ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('../input/movies-list/movies.list',sep='\t', names=col_names)
df.head()
mov_title=pd.read_csv('../input/movie-id-titles/Movie_Id_Titles')
mov_title.head()
df.loc[df['item_id']==50]
mov_title.loc[mov_title['item_id']==50]
df = pd.merge(df,mov_title,on = 'item_id')
df.head()
sns.set_style('white')
df.groupby('title')['rating'].mean().sort_values(ascending=False)
#note : some movies may have been rated only by one or two ppl which is why it tops in the below list. 
df.loc[df['title']=='Prefontaine (1997)']['rating'].count()
#commands practise - verifying mean value for one movie title
df.loc[df['title']=='1-900 (1994)']['rating'].mean()
# listing most rated movies based on the count(no. of times the movie was rated)
df.groupby('title')['rating'].count().sort_values(ascending=False)
# creating a dataframe for mean,ratings
ratings=pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
ratings['num of ratings'] = df.groupby('title')['rating'].count()
ratings.head()
ratings.head().sort_values(ascending=False,by ='num of ratings')
ratings.loc['Star Wars (1977)']
ratings['num of ratings'].hist(bins=50)
ratings['rating'].hist(bins=50)
sns.jointplot(x='rating',y='num of ratings',data=ratings,color='Purple')
df.head()
# create a matrix using a pivot
mov_mat=df.pivot_table(index='user_id', columns='title')
mov_mat.head()
mov_mat=df.pivot_table(index='user_id', columns='title',values='rating')
mov_mat.head()
ratings.head()
ratings.sort_values('num of ratings',ascending=False).head(10)
#choose a few movies and grab the rating
starwars_user_rating=mov_mat['Star Wars (1977)']
LiarLiar_user_rating=mov_mat['Liar Liar (1997)']
#find the correlartion bet one movie and the other movies

mov_mat.corrwith(starwars_user_rating)



similar_to_starwars=mov_mat.corrwith(starwars_user_rating)
corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.sort_values('correlation',ascending=False).head(10)
# most of the highly correlated values(ie similar to Star wars might hvae  been a less popular ones.
#The reason why it tops the list is probably because the movies were rated by less no. of people.
#it does not make sense to showcase these movies as recommendation for starwars. 
#So lets include the num. of ratings and set a threshold to aviod too many results)
corr_starwars =corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()
corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending=False).head()
LiarLiar_user_rating.head()
mov_mat.corrwith(LiarLiar_user_rating)

similar_to_liarliar=pd.DataFrame(mov_mat.corrwith(LiarLiar_user_rating),columns=['Correlation'])

similar_to_liarliar.dropna(inplace=True)
similar_to_liarliar.head()
similar_to_liarliar=similar_to_liarliar.join(ratings['num of ratings'])
similar_to_liarliar.head()
similar_to_liarliar[similar_to_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()


