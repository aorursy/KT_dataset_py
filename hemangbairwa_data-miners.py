# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Get the data 

column_names = ['user_id', 'item_id', 'rating', 'timestamp'] 



path = '../input/movie-recimendation/file.tsv'



df = pd.read_csv(path, sep='\t', names=column_names) 



# Check the head of the data 

df.head() 

# Check out all the movies and their respective IDs 

movie_titles = pd.read_csv('../input/movie-recimendation/Movie_Id_Titles.csv') 

movie_titles.head()
data = pd.merge(df, movie_titles, on='item_id') 

data.head()
data.groupby('title')['rating'].mean().head()
# Calculate mean rating of all movies 

data.groupby('title')['rating'].mean().sort_values(ascending=False).head() 
# Calculate count rating of all movies 

data.groupby('title')['rating'].count().sort_values(ascending=False).head() 
# creating dataframe with 'rating' count values 

ratings = pd.DataFrame(data.groupby('title')['rating'].mean()) 



ratings['num of ratings'] = pd.DataFrame(data.groupby('title')['rating'].count()) 



ratings.head()
import matplotlib.pyplot as plt 

import seaborn as sns 



sns.set_style('white') 

%matplotlib inline
# plot graph of 'num of ratings column' 

plt.figure(figsize =(10, 4)) 



ratings['num of ratings'].hist(bins = 70) 
# plot graph of 'ratings' column 

plt.figure(figsize =(10, 4)) 



ratings['rating'].hist(bins = 70) 
data = pd.merge(df, movie_titles, on='item_id') 

data.head() 
# Sorting values according to 

# the 'num of rating column' 

moviemat = data.pivot_table(index ='user_id', 

			columns ='title', values ='rating') 



moviemat.head() 



ratings.sort_values('num of ratings', ascending = False).head(10) 
# analysing correlation with similar movies 

starwars_user_ratings = moviemat['Star Wars (1977)'] 

liarliar_user_ratings = moviemat['Liar Liar (1997)'] 



starwars_user_ratings.head()
# analysing correlation with similar movies 

similar_to_starwars = moviemat.corrwith(starwars_user_ratings) 

similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings) 



corr_starwars = pd.DataFrame(similar_to_starwars, columns =['Correlation']) 

corr_starwars.dropna(inplace = True) 



corr_starwars.head()
# Similar movies like starwars 

corr_starwars.sort_values('Correlation', ascending = False).head(10) 

corr_starwars = corr_starwars.join(ratings['num of ratings']) 

corr_starwars.head()



corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending = False).head() 

# Similar movies as of liarliar 

corr_liarliar = pd.DataFrame(similar_to_liarliar, columns =['Correlation']) 

corr_liarliar.dropna(inplace = True) 



corr_liarliar = corr_liarliar.join(ratings['num of ratings']) 

corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation', ascending = False).head() 