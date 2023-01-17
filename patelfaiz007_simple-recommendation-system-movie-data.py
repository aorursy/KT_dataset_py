# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('../input/simple-recommendation-system/u.data', sep='\t', names=column_names)
df.head()
# Now let's get the movie titles:
movie_titles = pd.read_csv("../input/movie-titles/Movie_Id_Titles")
movie_titles.head()
# We can merge them together:
df = pd.merge(df,movie_titles,on='item_id')
df.head()
# EDA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline
# Let's create a ratings dataframe with average rating and number of ratings:
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
df.groupby('title')['rating'].count().sort_values(ascending=False).head()
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()
# Now a few histograms:
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
# Okay! Now that we have a general idea of what the data looks like, let's move on to creating a simple recommendation system:
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()
ratings.sort_values('num of ratings',ascending=False).head(10)
# Let's choose two movies: STARWARS and LIAR LIAR.
ratings.head()
# Now let's grab the user ratings for those two movies:
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()
# We can then use corrwith() method to get correlations between two pandas series:
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
# Let's clean this by removing NaN values and using a DataFrame instead of a series:
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()
# Now if we sort the dataframe by correlation, we should get the most similar movies, however note that we get some results that don't really make sense.
#This is because there are a lot of movies only watched once by users who also watched star wars (it was the most popular movie).
corr_starwars.sort_values('Correlation',ascending=False).head(10)
# Let's fix this by filtering out movies that have less than 100 reviews (this value was chosen based off the histogram from earlier). # We can try different values
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()
# Now sort the values and notice how the titles make a lot more sense:
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()
# Now the same for the Liar Liar:
corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()