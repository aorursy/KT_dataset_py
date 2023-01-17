# Supress Warnings
import warnings
warnings.filterwarnings('ignore')
# Import the numpy and pandas packages
import numpy as np
import pandas as pd
# Setting working directory to required location
import os
print(os.listdir("../input"))
movies = pd.DataFrame(pd.read_csv("../input/MovieAssignmentData.csv"))
movies.head()
# Row, columns in the Data
movies.shape
# Concise summary of the Dataframe.
movies.info()
# Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution,
# excluding NaN values
movies.describe()
movies.color.describe()
movies.director_name.describe()
movies.color.describe()
movies.genres.describe()
movies.movie_title.describe()
movies.director_name.describe()
movies.actor_3_name.describe()
movies.plot_keywords.describe()
movies.movie_imdb_link.describe()
movies.language.describe()
movies.country.describe()
movies.content_rating.describe()
# Write your code for column-wise null count here
movies.isnull().sum()
# Write your code for row-wise null count here
movies.isnull().sum(axis = 1)
# Write your code for column-wise null percentages here
round((movies.isnull().sum()/len(movies.index))*100,2)
# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations
movies = movies.drop(["color","director_facebook_likes","actor_1_facebook_likes","actor_2_facebook_likes","actor_3_facebook_likes","actor_2_name","cast_total_facebook_likes","actor_3_name","duration","facenumber_in_poster","content_rating","country","movie_imdb_link","aspect_ratio","plot_keywords"],axis =1)
movies.head()
movies.columns
# Write your code for dropping the rows here
round((movies.isnull().sum()/len(movies.index))*100,2)
movies = movies[~np.isnan(movies['gross'])]
movies = movies[~np.isnan(movies['budget'])]
# Write your code for dropping the rows here
movies = movies[movies.isnull().sum(axis=1) <= 5]
# Fill the NaN values in the 'language' column here
movies["language"] = movies.language.fillna("English")
movies.language.isnull().sum()
# Write your code for checking number of retained rows here
len(movies.index)/5043

# Write your code for unit conversion here\
movies.budget = round(movies["budget"]/1000000,2)
movies.gross = round(movies["gross"]/1000000,2)
# Write your code for creating the profit column here
movies["profit"] = movies.gross - movies.budget
# Write your code for sorting the dataframe here
movies_by_profit = movies.sort_values("profit",ascending = False)
top10 = movies_by_profit.head(10)
top10
# Write your code for dropping duplicate values here
movies = movies.drop_duplicates()
# Write code for repeating subtask 2 here
top10 = movies.head(10)
top10
# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 
# and name that dataframe as 'IMDb_Top_250'
IMDb_Top_250 = movies[movies["num_voted_users"] > 25000]
IMDb_Top_250 = IMDb_Top_250.sort_values("imdb_score", ascending = False).head(250)
IMDb_Top_250["rank"] = list(range(1,251))
IMDb_Top_250


Top_Foreign_Lang_Film = IMDb_Top_250[IMDb_Top_250["language"] != "English"]
Top_Foreign_Lang_Film.head()
# Write your code for extracting the top 10 directors here
director = movies.pivot_table(values = 'imdb_score', index = 'director_name', aggfunc = 'mean')
director = director.sort_values(by = 'imdb_score', ascending = False)
director = director.head(10)
director
# Write your code for extracting the first two genres of each movie here

movies["genre_1"] = movies["genres"].str.split("|").str.get(0)
movies["genre_2"] = movies["genres"].str.split("|").str.get(1)
movies["genre_2"] = movies["genre_2"].fillna(movies["genre_1"])
movies.head()
# Write your code for grouping the dataframe here
movies_by_segment = movies.groupby(['genre_1','genre_2'])
# Write your code for getting the 5 most popular combo of genres here
PopGenre = pd.DataFrame(movies_by_segment.gross.mean()).sort_values("gross", ascending = False).head()
PopGenre

# Write your code for creating three new dataframes here

Meryl_Streep = movies[movies["actor_1_name"] == "Meryl Streep"]
Meryl_Streep
Leo_Caprio = movies[movies["actor_1_name"] == "Leonardo DiCaprio"]
Leo_Caprio
Brad_Pitt = movies[movies["actor_1_name"] == "Brad Pitt"]
Brad_Pitt.head()
# Write your code for combining the three dataframes here
combined = pd.concat([Meryl_Streep,Leo_Caprio,Brad_Pitt ], axis = 0)
combined

# Write your code for grouping the combined dataframe here
movies_by_lead = combined.groupby('actor_1_name')
# Write the code for finding the mean of critic reviews and audience reviews here
print(movies_by_lead["num_critic_for_reviews"].mean())
print(movies_by_lead["num_user_for_reviews"].mean())
