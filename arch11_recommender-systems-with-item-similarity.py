
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv("/kaggle/input/movielens-data/u.data",sep="\t",names=column_names)
df.head()
movie_titles = pd.read_csv("/kaggle/input/movielens-data/Movie_Id_Titles.csv")
movie_titles.head()
print(df.shape,movie_titles.shape)
df = pd.merge(df,movie_titles,on="item_id")
df.shape
df.head()
df.groupby("title").mean()["rating"].sort_values(ascending=False)
df.groupby("title").count()["rating"].sort_values(ascending=False)
ratings = pd.DataFrame(df.groupby("title").mean()["rating"])
ratings
ratings["num of ratings"] = pd.DataFrame(df.groupby("title").count()["rating"])
ratings.head()
ratings["num of ratings"].hist(bins=50)
ratings["rating"].hist(bins=50)

sns.jointplot(x="rating",y="num of ratings",data=ratings,alpha=0.5)

df.head()
moviematrix = df.pivot_table(index="user_id",columns="title",values="rating")
moviematrix.head()
ratings.sort_values("num of ratings",ascending=False).head(10)
# choosing starwars and liarliar

starwars_user_ratings = moviematrix["Star Wars (1977)"]
liarliar_user_ratings= moviematrix["Liar Liar (1997)"]
starwars_user_ratings.head()
liarliar_user_ratings.head()
#correlation of ther movieswith starwars movie

similar_to_starwars = moviematrix.corrwith(starwars_user_ratings)
similar_to_liarliar = moviematrix.corrwith(liarliar_user_ratings)
starwars_corrdf = pd.DataFrame(similar_to_starwars,columns=["Correlation"])
starwars_corrdf= starwars_corrdf.join(ratings["num of ratings"])

liarliar_corrdf = pd.DataFrame(similar_to_liarliar,columns=["Correlation"])
liarliar_corrdf=liarliar_corrdf.join(ratings["num of ratings"])

starwars_corrdf.head()
starwars_corrdf[starwars_corrdf["num of ratings"] > 50].sort_values("Correlation",ascending=False).head() #filtering out movies with very less ratings ex: 100 reviews as threshold
liarliar_corrdf[liarliar_corrdf["num of ratings"] > 65].sort_values("Correlation",ascending=False).head() #filtering out movies with very less ratings ex: 100 reviews as threshold