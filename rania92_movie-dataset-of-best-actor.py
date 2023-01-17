# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
# Import the numpy and pandas packages



import numpy as np

import pandas as pd

import csv               #import the csv package
movies =pd.read_csv("../input/MovieAssignmentData.csv")     #importing the csv file

movies
movies.shape     #inspecting the number of rows and columns (rows, columns)
movies.info()    #inspecting the meta-data
movies.describe()      #statistical summary of the DataFrame
movies.isnull().sum()     #column-wise null count
movies.isnull().sum(axis=1)      #row-wise null count
round(100*(movies.isnull().sum()/len(movies.index)),2)       #column-wise null percentages
movies= movies.drop('color', axis=1)       #code for dropping the column 'color'

movies
movies= movies.drop('director_facebook_likes', axis=1)        #code for dropping the column 'director_facebook_likes'

movies
movies= movies.drop('actor_1_facebook_likes', axis=1)         #code for dropping the column 'actor_1_facebook_likes'

movies
movies= movies.drop('actor_2_facebook_likes', axis=1)         #code for dropping the column 'actor_2_facebook_likes'

movies
movies= movies.drop('actor_3_facebook_likes', axis=1)         #code for dropping the column 'actor_3_facebook_likes'

movies
movies= movies.drop('actor_2_name', axis=1)                   #code for dropping the column 'actor_2_name'

movies
movies= movies.drop('cast_total_facebook_likes', axis=1)      #code for dropping the column 'cast_total_facebook_likes'

movies
movies= movies.drop('actor_3_name', axis=1)                   #code for dropping the column 'actor_3_name'

movies
movies= movies.drop('duration', axis=1)                       #code for dropping the column 'duration'

movies
movies= movies.drop('facenumber_in_poster', axis=1)           #code for dropping the column 'facenumber_in_poster'

movies
movies= movies.drop('content_rating', axis=1)                 #code for dropping the column 'content_rating'

movies
movies= movies.drop('country', axis=1)                        #code for dropping the column 'country'

movies
movies= movies.drop('movie_imdb_link', axis=1)                #code for dropping the column 'movie_imdb_link'

movies
movies= movies.drop('aspect_ratio', axis=1)                  #code for dropping the column 'aspect_ratio'

movies
movies= movies.drop('plot_keywords', axis=1)                 #code for dropping the column 'plot_keywords'

movies
movies=movies[pd.notnull(movies['gross'])]      #code for dropping the rows which have null values for the column 'gross'
movies=movies[pd.notnull(movies['budget'])]     #code for dropping the rows which have null values for the column 'budget'
round(100*(movies.isnull().sum()/len(movies.index)),2)         #inspecting the column-wise null percentages
movies[movies.isnull().sum(axis=1)>5]         #code for dropping the rows having greater than five NaN values
movies.loc[pd.isnull(movies['language']), ['language']] = 'English'    #code for filling the NaN values in the 'language' column as English
movies.shape    #code for checking number of retained rows
                #code for checking the percentage of the rows retained
movies['budget']=round(movies.budget/1000000,2)        #code for unit conversion from $ to million $ in the column 'budget'
movies['gross']=round(movies.gross/1000000,2)          #code for unit conversion from $ to million $ in the column 'gross'
movies.rename(columns={'gross':'gross_in_millions'}, inplace=True)      #code for changing the column name

movies.rename(columns={'budget':'budget_in_millions'}, inplace=True)
movies['profit_in_millions'] = movies['gross_in_millions'] - movies['budget_in_millions']   #code for creating the profit column
movies.sort_values(by='profit_in_millions',ascending=False, inplace=True)   #code for sorting the dataframe with 'profit_in_millions' column as the reference
top10 = movies.nlargest(10, 'profit_in_millions')     #code to get the top 10 profiting movies

top10
top10.drop_duplicates(subset=None, keep='first', inplace=True)        #code for dropping duplicate values
movies.drop_duplicates(subset=None, keep='first', inplace=True)        #code for repeating subtask 2 after dropping duplicate values

movies.sort_values(by='profit_in_millions',ascending=False, inplace=True)

top10 = movies.nlargest(10, 'profit_in_millions')

top10
voted_users=movies.loc[movies.num_voted_users>25000, :]      #code for making sure 'num_voted_users' is greater than 25,000

voted_users.sort_values(by='imdb_score',ascending=False, inplace=True)

IMDb_Top_250 = voted_users.nlargest(250, 'imdb_score')      #code for extracting the top 250 movies as per the IMDb score 
initial_value=1          #code for creating the 'Rank' column

IMDb_Top_250['Rank'] = range(initial_value, len(IMDb_Top_250) +initial_value)

IMDb_Top_250 = IMDb_Top_250.set_index('Rank')

IMDb_Top_250
Top_Foreign_Lang_Film = IMDb_Top_250.loc[IMDb_Top_250.language!='English', :]        #code to extract top foreign language films from 'IMDb_Top_250'

Top_Foreign_Lang_Film
arrangebymean = movies.groupby('director_name', as_index=False)['imdb_score'].mean()

arrangebymean.sort_values(by='imdb_score',ascending=False, inplace=True)

top10directors = arrangebymean.nlargest(10, 'imdb_score')      #code for extracting the top 10 directors

top10directors
movies['genre_1'] = movies['genres'].str.split('|').str[0]      #code for extracting the first two genres of each movie

movies['genre_2'] = movies['genres'].str.split('|').str[1]

movies['genre_2'] = movies['genre_2'].fillna(movies['genre_1'])

movies
movies_by_segment = movies.groupby(['genre_1','genre_2'], as_index=False)['gross_in_millions'].mean()    #code for grouping the dataframe

movies_by_segment.sort_values(by='gross_in_millions',ascending=False, inplace=True)

movies_by_segment
PopGenre = movies_by_segment.nlargest(5, 'gross_in_millions')        #code for getting the 5 most popular combo of genres

PopGenre
Meryl_Streep = movies.loc[movies['actor_1_name'] == 'Meryl Streep', :]  #including all movies in which 'Meryl Streep' is the lead
Leo_Caprio = movies.loc[movies['actor_1_name'] == 'Leonardo DiCaprio', :]    #including all movies in which 'Leonardo DiCaprio' is the lead
Brad_Pitt = movies.loc[movies['actor_1_name'] == 'Brad Pitt', :]    #including all movies in which 'Brad Pitt' is the lead
temp=Meryl_Streep.append(Leo_Caprio, ignore_index=True)  #code for combining the three dataframes

Combined = temp.append(Brad_Pitt, ignore_index=True)

Combined
pregrouped = Combined.groupby(['actor_1_name'])     #code for grouping the combined dataframe

Grouped = pregrouped['num_critic_for_reviews','num_user_for_reviews'].mean()

Grouped
Final = Grouped.sort_values(by=['num_critic_for_reviews','num_user_for_reviews'], ascending=False) #code for finding the mean of critic reviews and audience reviews

Final