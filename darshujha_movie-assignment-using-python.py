# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
# Import the numpy and pandas packages



import numpy as np

import pandas as pd
movies =pd.read_csv('../input/movie_dataset.csv') # Write your code for importing the csv file here

movies
# Write your code for inspection here

print(movies.shape)

print(type(movies.info()))
# Write your code for column-wise null count here

movies.isnull().sum(axis=0)
# Write your code for row-wise null count here

movies.isnull().sum(axis=1)
# Write your code for column-wise null percentages here

round(100*(movies.isnull().sum()/len(movies.index)),2)
# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 

movies=movies.drop('color', axis=1)

movies=movies.drop('director_facebook_likes', axis=1)

movies=movies.drop('actor_1_facebook_likes', axis=1)

movies=movies.drop('actor_2_facebook_likes', axis=1)

movies=movies.drop('actor_3_facebook_likes', axis=1)

movies=movies.drop('actor_2_name', axis=1)

movies=movies.drop('cast_total_facebook_likes', axis=1)

movies=movies.drop('actor_3_name', axis=1)

movies=movies.drop('duration', axis=1)

movies=movies.drop('facenumber_in_poster', axis=1)

movies=movies.drop('content_rating', axis=1)

movies=movies.drop('country', axis=1)

movies=movies.drop('movie_imdb_link', axis=1)

movies=movies.drop('aspect_ratio', axis=1)

movies=movies.drop('plot_keywords', axis=1)

round(100*(movies.isnull().sum()/len(movies.index)),2)
# Write your code for dropping the rows here

movies = movies[~np.isnan(movies['gross'])]

movies = movies[~np.isnan(movies['budget'])]

round(100*(movies.isnull().sum()/len(movies.index)),2)
# Write your code for dropping the rows here

movies = movies[movies.isnull().sum(axis=1) <= 5]

round(100*(movies.isnull().sum()/len(movies.index)),2)
# Write your code for filling the NaN values in the 'language' column here

movies.loc[pd.isnull(movies['language']), ['language']] = 'English'

round(100*(movies.isnull().sum()/len(movies.index)), 2)
# Write your code for checking number of retained rows here

print(movies.index)

round(100*(len(movies.index)/5043),2)
# Write your code for unit conversion here

movies[['gross','budget']].apply(lambda x:x/1000000)
# Write your code for creating the profit column here

movies['profit']=movies['gross'] - movies['budget']

movies
# Write your code for sorting the dataframe here

movies.sort_values(by='profit',ascending=True)
top10=movies.sort_values(by=['profit'],ascending=False).head(10) # Write your code to get the top 10 profiting movies here

top10
# Write your code for dropping duplicate values here

movies.drop_duplicates()
# Write code for repeating subtask 2 here

top10=movies.sort_values(by=['profit'],ascending=False).head(10)

top10
# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 

# and name that dataframe as 'IMDb_Top_250'

IMDb_Top_250= movies.sort_values(by=['imdb_score'],ascending=False).head(250) # stored top 250 movies with highest imdb rating in new dataframe IMDb_top_250

IMDb_Top_250.loc[(IMDb_Top_250.num_voted_users>25),:]   # checking for all these movies,num_voted_users is greater than 25000

IMDb_Top_250['Rank']=range(1,251)      # Add a new column 'Rank' ranging from 1 to 251 indicate the ranks of the corresponding films  

IMDb_Top_250
Top_Foreign_Lang_Film =IMDb_Top_250.loc[(IMDb_Top_250.language != 'English'),:]       # Write your code to extract top foreign language films from 'IMDb_Top_250' here

Top_Foreign_Lang_Film 
# Write your code for extracting the top 10 directors here

top10director=movies.groupby('director_name')['imdb_score'].mean().sort_values(ascending=False).head(10)

top10director
# Write your code for extracting the first two genres of each movie here

first=movies['genres'].apply(lambda x: pd.Series(x.split('|')))

movies['genre_1']=first[0]

movies['genre_2']=first[1]

movies.loc[pd.isnull(movies['genre_2']), ['genre_2']] = movies['genre_1']

print(movies.genre_1)

print(movies.genre_2)
movies_by_segment =movies.groupby(['genre_1','genre_2']) # Write your code for grouping the dataframe here

movies_by_segment
PopGenre =movies_by_segment['gross'].mean().sort_values(ascending=False).head(5) # Write your code for getting the 5 most popular combo of genres here

PopGenre
# Write your code for creating three new dataframes here

Meryl_Streep=movies.loc[(movies.actor_1_name=='Meryl Streep'),:].head(3891)  # Include all movies in which Meryl_Streep is the lead

Meryl_Streep
Leo_Caprio=movies.loc[(movies.actor_1_name=='Leonardo DiCaprio'),:].head(3891) # Include all movies in which Leo_Caprio is the lead

Leo_Caprio
Brad_Pitt=movies.loc[(movies.actor_1_name=='Brad Pitt'),:].head(3891)  # Include all movies in which Brad_Pitt is the lead

Brad_Pitt
# Write your code for combining the three dataframes here

Combined=Meryl_Streep.append(Leo_Caprio).append(Brad_Pitt)

Combined
# Write your code for grouping the combined dataframe here

actor_name=Combined.groupby('actor_1_name')   # grouping the Combined dataframe using actor_1_name and store it in new dataframe actor_name

actor_name
# Write the code for finding the mean of critic reviews and audience reviews here

critic_reviews=actor_name['num_critic_for_reviews'].mean().sort_values(ascending=False).head(49)

print(critic_reviews)

audience_reviews=actor_name['num_user_for_reviews'].mean().sort_values(ascending=False).head(49)

print(audience_reviews)