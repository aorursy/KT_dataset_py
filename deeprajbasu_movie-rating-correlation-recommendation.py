# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#gathering the data 

column_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('/kaggle/input/ml100katoz/u.data', sep='\t', names=column_names)



df_titles= pd.read_csv('/kaggle/input/movie-id-title/Movie_Id_Titles')



df.head(10),df_titles.head(10)
df = pd.merge(df,df_titles,on='item_id')

df.head(10)
import matplotlib.pyplot as plt

%matplotlib inline
#number of movies in the data 

#number of total entries of movies

df['title'].nunique(),df['title'].count()



#looking at average rating of movies

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
#looking at number of individual ratings present by using the count method

df.groupby('title')['rating'].count().sort_values(ascending=False).head()
#creating a dataframe with above mentioned data

ratings_df = pd.DataFrame(df.groupby('title')['rating'].mean())

ratings_df['rating_count'] = df.groupby('title')['rating'].count()



ratings_df.head(10)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style='darkgrid')

#ns.set_palette("seismic", 25)





plt.figure(figsize=(10, 4))

sns.distplot(ratings_df['rating_count'],bins=35,).set_title("disstribution of number of ratings")

plt.figure(figsize=(10, 4))

sns.distplot(ratings_df['rating'],bins=150).set_title("disstribution of number of mean ratings")
sns.jointplot(x='rating',y='rating_count',data=ratings_df,sizes = (0.7,10)).fig.set_size_inches(8.25,7.65)
moviemap = df.pivot_table(index = 'user_id',columns = 'title',values='rating')

moviemap.head()





#by having the rating data for each movie in a separate column, it will be very easy

#for us to generate the corelation scores for each movie.
ratings_df.sort_values("rating_count",ascending=False).head(25)
#extracting the list of all the ratings for fargo movie by each user, 

#basically, how have all the users rated our chosen movie, we want to compare this

#to how our users have rated other movies, 



#many nan values because all the users who havent seen or rated this movie 



fargo_ratings = moviemap['Godfather, The (1972)']

fargo_ratings.head()
#corelating fargo ratings with every other film to get a correlation value 

fargo_similar = moviemap.corrwith(fargo_ratings)

fargo_similar[55:65]



#arranging this data into a dataframe 

fargo_corr = pd.DataFrame(fargo_similar,columns=['corr'])

fargo_corr.dropna(inplace=True)

fargo_corr.sort_values("corr",ascending=False,inplace=True)



fargo_corr.head()
fargo_corr=fargo_corr.join(ratings_df['rating_count'])



fargo_corr.head()

df =fargo_corr[fargo_corr['rating_count']>100]



df.sort_values("corr",ascending=False,inplace=True)



df.head()