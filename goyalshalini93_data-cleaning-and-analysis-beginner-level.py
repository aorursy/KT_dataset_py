# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
# Import the numpy and pandas packages



import numpy as np

import pandas as pd
movies = pd.read_csv("../input/MovieAssignmentData.csv") 

movies.head()
# Code for inspection here

print(movies.columns)
#Printing the dimensions of the movies dataframe

print(movies.shape)
# Looking at the datatypes of each column

print(movies.info())
# Code for column-wise null count here

movies.isnull().sum(axis=0)
# Code for row-wise null count here

movies.isnull().sum(axis=1)
# Code for column-wise null percentages here

print(round(100*(movies.isnull().sum()/len(movies.index)), 2))
# Code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 

movies = movies.drop(['color','director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes',

                     'actor_2_name','cast_total_facebook_likes','actor_3_name','duration','facenumber_in_poster','content_rating',

                     'country','movie_imdb_link','aspect_ratio','plot_keywords'], axis=1)



#printing the datatype and dataframe information after dropping the unneccesary columns

print(movies.info())
#printing the column-wise null percentage for the remaining columns in the dataset

print(round(100*(movies.isnull().sum()/len(movies.index)), 2))
# Write your code for dropping the rows for columns having more that 5% null values here

movies = movies[~(movies['gross'].isnull() | movies['budget'].isnull())]



#printing the column-wise null percentage for the remaining columns in the dataset

print(round(100*(movies.isnull().sum()/len(movies.index)), 2))
#length of rows with more than 5 NaN values

print(len(movies[movies.isnull().sum(axis=1) > 5]))



#length of rows with less or equal to 5 NaN values

print(len(movies[movies.isnull().sum(axis=1) <= 5]))



# Code for dropping the rows here

movies = movies[movies.isnull().sum(axis=1) <= 5]



#printing the column-wise null percentage for the remaining columns in the dataset

print(round(100*(movies.isnull().sum()/len(movies.index)), 2))
#converting language column as a category

movies['language'] = movies['language'].astype('category')



#printing the value counts for the language column

print(movies['language'].value_counts())



#imputing the NaN values by English - top language 

movies.loc[movies['language'].isnull(), ['language']] = 'English'



#printing the null percentage column-wise again

print(round(100*(movies.isnull().sum()/len(movies.index)), 2))
# Code for checking number of retained rows here

print(len(movies.index))



#Percentage of retained rows

print(100*(len(movies.index)/5043))
# Code for unit conversion here

movies['gross'] = movies['gross']/1000000

movies['budget'] = movies['budget']/1000000

movies
#Code for creating the profit column here

movies['profit'] = movies['gross'] - movies['budget']

movies
# Code for sorting the dataframe here

movies.sort_values('profit',ascending = False)
# Code to get the top 10 profiting movies here

top10 = movies.sort_values('profit',ascending = False).head(10) 

top10
# Code for dropping duplicate values here

movies = movies[~movies.duplicated()]

movies
# Code for repeating subtask 2 here

top10 = movies.sort_values('profit',ascending = False).head(10)

top10
# Code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 

# and name that dataframe as 'IMDb_Top_250'

IMDb_Top_250 = movies.sort_values(by = 'imdb_score', ascending = False)

IMDb_Top_250 = IMDb_Top_250.loc[IMDb_Top_250.num_voted_users > 25000]

IMDb_Top_250 = IMDb_Top_250.iloc[:250, ]

IMDb_Top_250['Rank'] = range(1,251)

IMDb_Top_250
# Code to extract top foreign language films from 'IMDb_Top_250' here

Top_Foreign_Lang_Film = IMDb_Top_250[IMDb_Top_250['language'] != 'English']

Top_Foreign_Lang_Film
# Code for extracting the top 10 directors here

top10director = pd.DataFrame(movies.groupby('director_name').imdb_score.mean().sort_values(ascending = False).head(10))

top10director
# Code for extracting the first two genres of each movie here

movies['genre_1'] = movies['genres'].apply(lambda x: x.split('|')[0])

movies['genre_2'] = movies['genres'].apply(lambda x: x.split('|')[0] if len(x.split('|'))<2 else x.split('|')[1])

movies
# Code for grouping the dataframe here

movies_by_segment = movies.groupby(['genre_1','genre_2'])
# Code for getting the 5 most popular combo of genres here

PopGenre = pd.DataFrame(movies_by_segment.gross.mean().sort_values(ascending = False).head(5))

PopGenre
# Code for creating three new dataframes here

# Including all movies in which Meryl_Streep is the lead

Meryl_Streep = movies.loc[movies['actor_1_name'] == 'Meryl Streep']

Meryl_Streep
# Including all movies in which Leo_Caprio is the lead

Leo_Caprio = movies.loc[movies['actor_1_name'] == 'Leonardo DiCaprio']

Leo_Caprio
# Including all movies in which Brad_Pitt is the lead

Brad_Pitt = movies.loc[movies['actor_1_name'] == 'Brad Pitt']

Brad_Pitt
# Code for combining the three dataframes here

combined = Meryl_Streep.append(Leo_Caprio.append(Brad_Pitt))

combined
# Code for grouping the combined dataframe here

combined = combined.groupby('actor_1_name')
# Code for finding the mean of critic reviews and audience reviews here

combined['num_critic_for_reviews','num_user_for_reviews'].mean().sort_values('num_critic_for_reviews',ascending = False)