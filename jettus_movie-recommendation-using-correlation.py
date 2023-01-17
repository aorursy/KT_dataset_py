import pandas as pd

import numpy as np
column_names = ['user_id','item_id','rating','timestamp']

df = pd.read_csv("../input/u.data.txt",sep='\t',names = column_names)
df.head()
#movies 
col_names = ['item_id','title']

df_movies = pd.read_csv('../input/movies.txt',sep=',')
df_movies.head()
#merging the two datasets using the 'item_id' which is common in both the datasets

df=pd.merge(df,df_movies,on='item_id')
df.head()
from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('white') 
df.groupby('title')['rating'].count().sort_values(ascending=False).head()


ratings =  pd.DataFrame(df.groupby('title')['rating'].mean())

ratings.head()
#making a seperate column for number of rating

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

ratings.head()
#ploting the distribution of number of rating

plt.figure(figsize=(10,4))

ratings['num of ratings'].hist(bins=70)
plt.figure(figsize=(10,4))

ratings['rating'].hist(bins=70)
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
moviemat = df.pivot_table(index='user_id',columns ='title',values ='rating')

moviemat.head()
ratings.sort_values('num of ratings',ascending=False).head(10)
starwars_user_ratings = moviemat['Star Wars (1977)']

liarliar_user_ratings = moviemat['Liar Liar (1997)']

starwars_user_ratings.head()
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)

similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
corr_starwars = pd.DataFrame(similar_to_starwars,columns =['Correlation'])

corr_starwars.dropna(inplace=True)

corr_starwars.head()
corr_starwars.sort_values('Correlation',ascending=False).head(10)
corr_starwars= corr_starwars.join(ratings['num of ratings'])

corr_starwars.head()
#final step 

corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()
#lets do the same for the another movie and get recomendation for that too

corr_liarliar = pd.DataFrame(similar_to_liarliar,columns =['Correlation'])

corr_liarliar.dropna(inplace=True)

corr_liarliar.sort_values('Correlation',ascending=False).head(10)
corr_liarliar= corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()