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
movies = pd.read_csv("../input/movies_metadata.csv");

ratings = pd.read_csv("../input/ratings_small.csv");

keywords_plot = pd.read_csv("../input/keywords.csv");

credits = pd.read_csv("../input/credits.csv");

rate = pd.read_csv("../input/ratings.csv");

links = pd.read_csv("../input/links.csv");
#let's vizualise MOVIES Data Frame and clean this data set to find Top 3 Revenue Movies.

movies
#Column-Wise

movies.isnull().sum()
# checking the percentage of null values

round(100*(movies.isnull().sum()/len(movies.index)),2)
# We wiil drop the Columns which have null values more than 0.1%

## These Columns are : 'belongs_to_collection','homepage','overview','release_date','tagline','runtime','status'

# We will even drop Poster Path and Overview as these are not required columns

movies = movies.drop(['belongs_to_collection','homepage','overview','release_date','tagline','poster_path','overview','runtime','status'],axis=1)

movies
# there might be some null values still, so let us check the % of null values present still

round(100*(movies.isnull().sum()/len(movies.index)),2)
movies.shape
# Now we will drop the NULL Values in the Rows

movies = movies[~np.isnan(movies.revenue)]

movies
movies.shape
round(100*(movies.isnull().sum()/len(movies.index)),2)
movies = movies[movies.isnull().sum(axis=1)<=5]

movies
movies.original_language.describe()
# So we set all the missing values in the Data frame with en

movies.loc[pd.isnull(movies['original_language']),['original_language']] = 'en'

movies
movies.isnull().sum()
movies.imdb_id.describe()
#we replace the NULL Values with the most frequent 'tt1180333', but we should not do that. So either we drop that column or we replace it with most frequent ID

#movies.loc[pd.isnull(movies['imdb_id']),['imdb_id']] = 'tt1180333'

#movies

#Instead of Imputing Values, we try to delete the Rows where imdb_id is a Null Value

movies = movies[movies['imdb_id'].notnull()]
movies
# Now let's check wether we have any null value or not

movies.isnull().sum()
# % for null values

round(100*(movies.isnull().sum()/len(movies.index)),2)
# Now the Data is Clean of any Missing Values.

# So, Top 3 revenues movies:

movies = movies.sort_values(by = 'revenue',ascending=False)

movies
#Top 3 Revenue Movie

top3revenue = movies.loc[:3,]

top3revenue


movies.drop_duplicates(subset = None,keep='first',inplace=True)

movies

movies.set_index(['id'])
#Top 10 movies according to Revenue.

movies.iloc[:10,]
ratings
ratings.isnull().sum()

#We see that It is one of the cleaned data frames.

#Next Step we think of is to Merge/Concat the Data Frame. What do you think....should we Merge or Concat??
top10 = ratings.sort_values(by='rating',ascending=False)

top10 = top10.iloc[:10,]

top10
#top 10 movies

top10 = top10.drop(['timestamp'],axis=1)

top10
#Now let's dive into other Data Frames too!

keywords_plot
keywords_plot.isnull().sum()
round(100*(keywords_plot.isnull().sum()/len(keywords_plot.index)),2)
#Let's split the data and Apply a ML Model to predict the imdb_score for the testing data!
movies.drop(['vote_count'],axis=1)
X = movies.loc[:,:'video'].as_matrix()

y = movies['vote_average']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

print(X_train.shape,X_test.shape)

print(y_train.shape,y_test.shape)