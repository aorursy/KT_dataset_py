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

        

# Supress Warnings



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
import pandas as pd



# code for importing the csv file

movies = pd.read_csv("../input/MovieAssignmentData.csv")

movies.head()
# code for inspection here

# Column Names

movies.columns
# The number of rows and columns

n_rows_org = movies.shape[0]  #Storing this value as it is needed in Subtask 2.6

print("Number of Rows: {0} \nNumber of Columns: {1}".format(movies.shape[0],movies.shape[1]))
# Looking at the top and bottom entries of dataframes

print(movies.head())

print("\n\n ---------------------------------------------------- \n\n")

print(movies.tail())
# Looking at the datatypes of each column (variable types)

movies.info()
# Summary of all the numeric columns in the dataset

movies.describe()
# code for column-wise null count

print(movies.isnull().sum())
# code for row-wise null count

print(movies.isnull().sum(axis=1))
# code for column-wise null percentages

print(round(movies.isnull().sum()/len(movies.index)*100, 2))
# code for dropping the columns 

movies.drop(['color',

             'director_facebook_likes',

             'actor_1_facebook_likes',

             'actor_2_facebook_likes',

             'actor_3_facebook_likes',

             'actor_2_name',

             'cast_total_facebook_likes',

             'actor_3_name',

             'duration',

             'facenumber_in_poster',

             'content_rating',

             'country',

             'movie_imdb_link',

             'aspect_ratio',

             'plot_keywords'], axis=1, inplace=True)
#Inspecting that drop operation has been perfomed properly

movies.head(3)
#Percentage of null values in all columns

#Getting list of columns having null value percentage > 5

drop_list  = list(movies.columns[(round(movies.isnull().sum(axis=0)/len(movies.index)*100, 2) > 5)])

drop_list
# code for dropping the rows

# Since gross and budget columns are having greater the 5% Null values,

# we will drop respective rows with null values in these two columns



# movies = movies[pd.notnull(movies['gross']) & pd.notnull(movies['budget'])]

movies = movies.dropna(subset = drop_list)
# Rechecking Percentage of null values in all columns

print(round(movies.isnull().sum()/len(movies.index)*100, 2))
# code for dropping the rows 

movies = movies[movies.isnull().sum(axis=1) < 5]
# code for filling the NaN values in the 'language' column

movies['language'] = movies['language'].fillna('English')
# code for checking number of retained rows

print("Number of Retaind Rows: {0} \nNumber of Retained Columns: {1}".format(movies.shape[0],movies.shape[1]))
# percentage of the rows retained 

print((len(movies.index)/n_rows_org)*100)
# code for unit conversion

# To Convert budget unit to million $ we need to divide it by 1000000. 

movies['budget'] = round((movies['budget']/1000000), 2)

movies['gross'] = round((movies['gross']/1000000), 2)
# code for creating the profit column

movies['profit'] = movies['gross'] - movies['budget']
# code for sorting the dataframe

movies.sort_values(by = 'profit', ascending = False)
# code to get the top 10 profiting movies

top10 = movies.sort_values(by = 'profit', ascending = False).head(10).reset_index(drop=True)

top10
# code for dropping duplicate values

movies.drop_duplicates(subset=['movie_title'], inplace = True)
# code for repeating subtask 2

top10 = movies.sort_values(by = 'profit', ascending = False).head(10).reset_index(drop=True)

top10
# code for extracting the top 250 movies as per the IMDb score. 

#  we store it in a new dataframe and name that dataframe as 'IMDb_Top_250'

IMDb_Top_250 = movies[movies.num_voted_users > 25000].sort_values(by = 'imdb_score', ascending = False).head(250)

IMDb_Top_250['Rank'] = list(range(1, 251))

IMDb_Top_250.head()
# code to extract top foreign language films from 'IMDb_Top_250'

Top_Foreign_Lang_Film = IMDb_Top_250[IMDb_Top_250['language'] != 'English']

Top_Foreign_Lang_Film
# code for extracting the top 10 directors here

# Creatingg grouby object using director_name

grp_1 = movies.groupby('director_name')

temp = grp_1['imdb_score'].mean().sort_values(ascending = False)



# Storing result of groupby aggregation into dataframe

top10director = pd.DataFrame({'mean_score' : temp}).reset_index().head(10)

top10director
# code for extracting the first two genres of each movie

temp1 = movies['genres'].str.split(pat='|', expand=True)

movies['genre_1'] = temp1[0]

movies['genre_2'] = temp1[1]
#filling null value in genre_2 using genre_1 values

movies['genre_2'] =  movies['genre_2'].fillna(movies['genre_1'])
# code for grouping the dataframe by genre_1 and genre_2

movies_by_segment = movies.groupby(['genre_1', 'genre_2'])
# code for getting the 5 most popular combo of genres

PopGenre = pd.DataFrame({'Total_gross' : movies_by_segment['gross'].mean()}).reset_index()

PopGenre = PopGenre.sort_values(by = 'Total_gross', ascending = False).head(5)

PopGenre
# code for creating three new dataframes

# Include all movies in which Meryl_Streep is the lead

Meryl_Streep = movies[movies['actor_1_name'] == 'Meryl Streep']
# Include all movies in which Leo_Caprio is the lead 

Leo_Caprio = movies[movies['actor_1_name'] == 'Leonardo DiCaprio']
# Include all movies in which Brad_Pitt is the lead

Brad_Pitt =  movies[movies['actor_1_name'] == 'Brad Pitt'] 
# code for combining the three dataframes

Combined = pd.concat([Meryl_Streep, Leo_Caprio, Brad_Pitt], axis = 0)
# code for grouping the combined dataframe

temp3 = Combined.groupby('actor_1_name')
# code for finding the mean of critic reviews and audience reviews

temp3[['num_critic_for_reviews', 'num_user_for_reviews']].mean().sort_values(by = ['num_critic_for_reviews', 'num_user_for_reviews'], ascending = False)