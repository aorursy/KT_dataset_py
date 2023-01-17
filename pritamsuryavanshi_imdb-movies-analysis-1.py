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
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt   
import seaborn as sns
####  Step 1. Reading dataset and Data Inspection
movies = pd.read_csv("/kaggle/input/imdb-movies-analysis/MovieAssignmentData.csv")
movies
movies.info()
movies.shape
movies.describe()
movies.columns
####  Step 2. Data Analysis and Data Cleaning
# Column-wise Null count
movies.isnull().sum()
# row-wise null count here
movies.isnull().sum(axis=1) 
len(movies.index)
# Column-wise null percentage
round(100*(movies.isnull().sum(axis=0)/len(movies.index)),2)
# Here , dropping below unnecessary columns as we will mostly be analyzing the movies with respect to the ratings, gross collection, popularity of movies, etc. So many of the columns in this dataframe are not required. So it is advised to drop the following columns.

# color
# director_facebook_likes
# actor_1_facebook_likes
# actor_2_facebook_likes
# actor_3_facebook_likes
# actor_2_name
# cast_total_facebook_likes
# actor_3_name
# duration
# facenumber_in_poster
# content_rating
# country
# movie_imdb_link
# aspect_ratio
# plot_keywords
movies = movies.drop(['color', 
                      'director_facebook_likes', 
                      'actor_3_facebook_likes', 
                      'actor_1_facebook_likes', 
                      'cast_total_facebook_likes', 
                      'actor_2_facebook_likes', 
                      'duration', 
                      'facenumber_in_poster', 
                      'content_rating', 
                      'country', 
                      'movie_imdb_link', 
                      'aspect_ratio',
                      'plot_keywords',
                      'actor_2_name',
                      'actor_3_name'], 
                       axis = 1)
movies
# again to check Null percentage
round(100*(movies.isnull().sum()/len(movies.index)),2)
# Now, on inspection we notice that some columns have large percentage (greater than 5%) of Null values. Drop all the rows which have Null values for such columns.
movies[movies.isnull().sum(axis=1)>5]
# count the number of rows having > 5 missing values
len(movies[movies.isnull().sum(axis=1) > 5].index)
# 4 rows have more than 5 missing values
# calculate the percentage
100*(len(movies[movies.isnull().sum(axis=1) > 5].index) / len(movies.index))
# retaining the rows having <= 5 NaNs
movies = movies[movies.isnull().sum(axis=1) <= 5]

# look at the summary again
round(100*(movies.isnull().sum()/len(movies.index)), 2)
movies = movies[~pd.isnull(movies['gross'])]  #As per above result we have gross and budget columns has greater than 5% of Null values
movies = movies[~pd.isnull(movies['budget'])]  #change this code as np.isnan not required.
movies
round(100*(movies.isnull().sum()/len(movies.index)), 2)  #check the Null percentage again
# The rows for which the sum of Null is less than five are retained
movies = movies[movies.isnull().sum(axis=1) <= 5]
movies
# Inspecting the percentages of NaN

round(100*(movies.isnull().sum()/len(movies.index)), 2)
# Inspect the language column of the dataset as Language column has NaN values, so  it is safe to replace all the missing values with 'English'
movies['language'].describe()
# Fill the NaN values with 'English' since most of the movies are in the English language

movies.loc[pd.isnull(movies['language']), ['language']] = 'English'
movies
# Inspecting the percentages of NaNs

round(100*(movies.isnull().sum()/len(movies.index)), 2)
print(len(movies.index))
print(len(movies.index)/5042)
# Convert the unit of the budget and gross columns from $ to million $.
# Divide the 'gross' and 'budget' columns by 1000000 to convert '$' to 'million $'

movies['gross'] = movies['gross']/1000000
movies['budget'] = movies['budget']/1000000
movies
# Create the new column named 'profit' by subtracting the 'budget' column from the 'gross' column

movies['profit'] = movies['gross'] - movies['budget']
movies
# Sort the dataframe with the 'profit' column as reference using the 'sort_values' function. Make sure to set the argument
# 'ascending' to 'False'

movies = movies.sort_values(by = 'profit', ascending = False)
movies
# Get the top 10 profitable movies by using position based indexing. Specify the rows till 10 (0-9)

top10 = movies.iloc[:10, ]
top10
# Drop the duplicate values using 'drop_duplicates' function. All the columns for duplicate rows need to be dropped and thus,
# the 'subset' argument is set to 'None'. The 'keep = first' indicates to retain the first row among the duplicate rows, and
# 'inplace = True' performs the operation on the dataframe in place.

movies.drop_duplicates(subset = None, keep = 'first', inplace = True)
movies
# Get the top 10 profitable movies by using position based indexing. Specify the rows till 10 (0-9)

top10 = movies.iloc[:10, ]
top10
# Sort the movies by IMDb score
# Retain the movies with 'num_voted_users' greater than 25000
# Use position based indexing to get the first 250 rows in the sorted dataframe
# Create a new column rank which contains the rank from 1 to 250

IMDb_Top_250 = movies.sort_values(by = 'imdb_score', ascending = False)
IMDb_Top_250 = IMDb_Top_250.loc[IMDb_Top_250.num_voted_users > 25000]
IMDb_Top_250 = IMDb_Top_250.iloc[:250, ]
IMDb_Top_250['Rank'] = range(1,251)
IMDb_Top_250
# Get the non-English language films using conditional label based indexing

Top_Foreign_Lang_Film = IMDb_Top_250.loc[IMDb_Top_250['language'] != 'English']
Top_Foreign_Lang_Film
# Create a pivot table using 'director_name' as index, 'imdb_score' as values, and 'mean' as aggfunc
# Sort the values by 'imdb_score'. Keep 'ascending' as 'False'
# Extract the top 10 from the dataframe created

# PS: If I had to find the worst 10 directors, I would have sorted the dataframe in an ascending order and again extrcated the first 10 rows

director = movies.pivot_table(values = 'imdb_score', index = 'director_name', aggfunc = 'mean')
director = director.sort_values(by = 'imdb_score', ascending = False)
director = director.iloc[:10, ]
director
# Split the elements of the 'genre' column at the pipe characters ('|') using str.split()
# Assign the first elements of the rows of 'genre' column to a new column named 'genre_1' using 'apply()' and 'lambda' functions
# Some of the movies have only one genre. In such cases, assign the same genre to 'genre_2' as well

movies['genres'] = movies['genres'].str.split('|')
movies['genre_1'] = movies['genres'].apply(lambda x: x[0])
movies['genre_2'] = movies['genres'].apply(lambda x : x[1] if len(x) > 1 else x[0])
movies
# Group the dataframe using 'genre_1' as the primary column and 'genre_2' as secondary

movies_by_segment = movies.groupby(['genre_1', 'genre_2'])
# Create a new dataframe PopGenre which contains the 'mean' of the gross values of each combination of genres present
# Sort this dataframe using the 'gross' column and use index-based positioning to find out the five most popular genre combos

PopGenre = pd.DataFrame(movies_by_segment['gross'].mean()).sort_values(by = 'gross', ascending = False)
PopGenre.iloc[:5, ]
# Create a new dataframe containing Meryl Streep movies in which she is the lead actor

Meryl_Streep = movies.loc[movies.actor_1_name == 'Meryl Streep']
Meryl_Streep
# Create a new dataframe containing Leonardo DiCaprio movies in which he is the lead actor

Leo_Caprio = movies.loc[movies.actor_1_name == 'Leonardo DiCaprio']
Leo_Caprio
# Create a new dataframe containing Brad Pitt movies in which he is the lead actor

Brad_Pitt = movies.loc[movies.actor_1_name == 'Brad Pitt']
Brad_Pitt
# Combine the three dataframes using 'pd.concat()'

Combined = pd.concat([Meryl_Streep, Brad_Pitt, Leo_Caprio])
Combined
# Group the dataframe by 'actor_1_name'

Combined_by_segment = Combined.groupby('actor_1_name')
# Remember that we had some null values for the column 'num_critic_for_reviews'. Make sure that none of these null values are
# present in the new dataframe - 'Combined' that we have created

Combined.isnull().sum()
# Find the mean of 'num_user_for_reviews' for each of the actor. Notice, Leonardo's is the highest

Combined_by_segment['num_user_for_reviews'].mean()
# Find the mean of 'num_critic_for_reviews' for each of the actor. In this case as well, Leonardo is leading

Combined_by_segment['num_critic_for_reviews'].mean()
# Write the code for calculating decade here
movies['decade'] = movies['title_year']//10*10   #store title_year data in new dataframe decade
movies
 #find the sum of each decade and store them in df_by_decade new dataframe with desc order.
df_by_decade=movies.groupby('decade')['num_voted_users'].sum().sort_values(ascending=False)
df_by_decade  
# Write your code for plotting number of voted users vs decade

sns.set_style("white")
plt.figure(figsize=(12, 6))   # increase figure size 
sns.barplot(x ='decade', y ='num_voted_users', data = movies , palette = 'spring')
plt.yscale('log') #use log to observe distribution of plot in better way 
plt.show()        #outliers - decade 2000 has max num voted and decade 1920 has less. 
