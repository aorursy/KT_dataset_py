# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
# Import the numpy and pandas packages



import numpy as np

import pandas as pd
movies = pd.read_csv('../input/MovieAssignmentData.csv')

movies.head()
# Write your code for inspection here

movies.info()
# Checking the no of rows and columns

movies.shape
# Checking the spread of the values

movies.describe()
# Write your code for column-wise null count here

movies.isnull().sum(axis = 0)
# Write your code for row-wise null count here

movies.isnull().sum(axis = 1)
# Write your code for column-wise null percentages here

round(movies.isnull().sum()/len(movies.index)*100,2)
# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 

movies = movies.drop('color', axis=1)

# Checking the null percentages and this shows the remaining columns as well

round(movies.isnull().sum()/len(movies.index)*100,2)
# Since first drop was successful, dropping the other columns

movies = movies.drop(['director_facebook_likes',

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

                      'plot_keywords'], axis=1)

# Checking the null percentages and this shows the remaining columns as well

round(movies.isnull().sum()/len(movies.index)*100,2)
# Write your code for dropping the rows here

movies = movies.dropna(axis=0, subset=['gross', 'budget'])



# Checking the remaining percentages

round(movies.isnull().sum()/len(movies.index)*100,2)
# Write your code for filling the NaN values in the 'language' column here

movies['language'].fillna('English', inplace=True)



# Checking the remaining percentages

round(movies.isnull().sum()/len(movies.index)*100,2)
# Checking the language Column for any Nan values

movies['language'].isnull().sum()
# Write your code for checking number of retained rows here



# Checking no of rows now

movies.shape[0]
# We already know that the no of rows were 5043 at the beginning, thus calculating the percentage of remaining rows

round(movies.shape[0]/5043*100,2)
# Write your code for unit conversion here

def conv_to_million(x):

    return x/1000000



movies['budget'] = movies['budget'].apply(conv_to_million)

movies['gross'] = movies['gross'].apply(conv_to_million)



# Checking the result

movies.head()
# Write your code for creating the profit column here

movies['profit'] = movies['gross'] - movies['budget']

movies.head()
# Write your code for sorting the dataframe here

movies = movies.sort_values(['profit'])



# Checking the least profitable movies

movies.head()
# Write code for profit vs budget plot here

# I am interested to know if there is any co-relation between the budget of a movies and the profit it makes

# for that a scatter plot would give us a good idea of high budget means high profit or not

import matplotlib.pyplot as plt

import seaborn as sns



plt.scatter(movies['profit'],movies['budget'],facecolors = 'none', edgecolors = 'r')

plt.xlabel('Profit')

plt.ylabel('Budget')

plt.show()
# Since the Dataframe is sorted on profit, we already have the last 10 rows as the most profitable rows

top10 = movies.tail(10)

top10.sort_values("profit", inplace=True,ascending=False)   #Result requested in descending order 

top10
# Write your code for dropping duplicate values here



# Sorting by movie_title

movies.sort_values("movie_title", inplace=True) 

# Dropping the line of the full row is a duplicate

movies.drop_duplicates(keep='first',inplace=True) 

movies.shape[0]
# Write code for repeating subtask 2 here

movies.sort_values("profit", inplace=True)   #Putting the profitable movies at the end

top10 = movies.tail(10)

top10.sort_values("profit", inplace=True,ascending=False)   #Result requested in descending order 

top10
# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 

# and name that dataframe as 'IMDb_Top_250'



#First extracting all movies with num_voted_users is greater than 25,000

IMDb = movies.loc[movies['num_voted_users']>25000]

IMDb.shape
#Extracting the top 250 movies

IMDb_Top_250 = IMDb.nlargest(250, 'imdb_score', keep='all')                            

IMDb_Top_250.shape    #it is not necessary to have 250 lines since many movies can have equal IMDB Scores
Top_Foreign_Lang_Film = IMDb_Top_250.loc[IMDb_Top_250['language'] != 'English']

Top_Foreign_Lang_Film
# Checking for Veer Zara in Hindi movie section

temp = IMDb_Top_250.loc[(IMDb_Top_250.language=='Hindi',)]

temp.head()
# Write your code for extracting the top 10 directors here

director_list = movies.groupby('director_name').imdb_score.mean().sort_values(ascending=False).reset_index()

top10director = director_list.nlargest(10, 'imdb_score')

top10director
# Write your code for extracting the first two genres of each movie here

movies['genres'] = movies.genres.str.split('|')

movies['genre_1'] = movies['genres'].apply(lambda x: x[0])

movies['genre_2'] = movies['genres'].apply(lambda x: x[1] if len(x)>1 else x[0])

movies.head()
movies_by_segment = movies.groupby(['genre_1','genre_2'])

movies_by_segment.head()
PopGenre = movies_by_segment.gross.mean().sort_values(ascending=False).reset_index()

PopGenre.head()
# Write your code for creating three new dataframes here



Meryl_Streep = movies[movies['actor_1_name']=='Meryl Streep']

Meryl_Streep.head()
Leo_Caprio = movies[movies['actor_1_name']=='Leonardo DiCaprio']

Leo_Caprio.head()
Brad_Pitt = movies[movies['actor_1_name']=='Brad Pitt']

Brad_Pitt.head()
# Write your code for combining the three dataframes here

df = Meryl_Streep.append(Leo_Caprio)

Combined = df.append(Brad_Pitt)



Combined.head()
Combined.tail()
# Write your code for grouping the combined dataframe here

combined_group = Combined.groupby('actor_1_name')

combined_group.head()
# Write the code for finding the mean of critic reviews and audience reviews here

mean_critic = Combined.groupby('actor_1_name').num_critic_for_reviews.mean()

mean_user = Combined.groupby('actor_1_name').num_user_for_reviews.mean()

mean_critic.head()
mean_user.head()
# Write the code for calculating decade here

movies['decade'] = movies['title_year'].apply(lambda x: x//10 * 10).astype(np.int64)   #changing to int to avoid the .0

movies['decade'] = movies['decade'].astype(str) + 's'

movies = movies.sort_values(['decade'])

movies.head()
# Write your code for creating the data frame df_by_decade here

movies['decade'] = movies['decade'].astype(str)

df_by_decade = movies.groupby('decade')['num_voted_users'].sum().reset_index()

df_by_decade.head(10)
# Write your code for plotting number of voted users vs decade

import seaborn as sns

sns.set(style="white")

sns.barplot(x='decade',y='num_voted_users', data=df_by_decade)

plt.xlabel('Decades')

plt.ylabel('Number of Voted Users')

plt.show()