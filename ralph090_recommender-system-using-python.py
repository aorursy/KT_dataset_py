import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
column_names=['user_id','item_id','rating','timestamp']
df =pd.read_csv('../input/movielens/u.data.csv',sep='\t',names=column_names)
movieT=pd.read_csv('../input/movie-titles/Movie_Id_Titles')
df.head(6)
#Similarly,
movieT.head(5) #To check first 5 rows of the DataFrame.
merger=pd.merge(df,movieT,on='item_id')
merger.head(5)
merger.groupby('title')['rating'].mean().sort_values(ascending=False).head()
merger.groupby('title')['rating'].count().sort_values(ascending=False).head()
ratings = pd.DataFrame(merger.groupby('title')['rating'].mean())
ratings.head()
ratings['NumofRatings'] = pd.DataFrame(merger.groupby('title')['rating'].count())
ratings.head()
plt.figure(figsize=(10,4))
ratings['NumofRatings'].hist(bins=80,color='black') 
plt.title("Count of Rating given by the user")
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=50,color='black') 
plt.title("Rating provided by the user")
sns.jointplot(x='rating',y='NumofRatings',data=ratings,alpha=0.3,color='k',kind='scatter',marker="*",)
moviematrix = merger.pivot_table(index='user_id',columns='title',values='rating')
moviematrix.head(5)
ratings.sort_values('NumofRatings',ascending=False).head(10)
starwars_Uratings = moviematrix['Star Wars (1977)']
liarliar_Uratings = moviematrix['Liar Liar (1997)']
starwars_Uratings.head(), liarliar_Uratings.head(), print("The object type is: {}".format(type(starwars_Uratings))) #Returns a Pandas series object

similar_starwars = moviematrix.corrwith(starwars_Uratings)
similar_liarliar = moviematrix.corrwith(liarliar_Uratings)
corr1starwars = pd.DataFrame(similar_starwars,columns=['Correlation_StarWars'])
corr1starwars.dropna(inplace=True)
corr1starwars.head()
corr1starwars.sort_values('Correlation_StarWars',ascending=False).head(50)
corr1starwars = corr1starwars.join(ratings['NumofRatings'])
Recommendations_StarWars=corr1starwars[corr1starwars['NumofRatings']>100].sort_values('Correlation_StarWars',ascending=False).head(10)
print("Recommendations based on Star Wars (1977) movie are")
Recommendations_StarWars
corr1liarliar = pd.DataFrame(similar_liarliar,columns=['Correlation_liarliar'])
corr1starwars.dropna(inplace=True)
corr1liarliar.head()
corr1liarliar.sort_values('Correlation_liarliar',ascending=False).head(50)
corr1liarliar = corr1liarliar.join(ratings['NumofRatings'])
Recommendations_liarliar=corr1liarliar[corr1liarliar['NumofRatings']>100].sort_values('Correlation_liarliar',ascending=False).head(10)
print("Recommendations based on Liar Liar (1997) movie are:")
Recommendations_liarliar