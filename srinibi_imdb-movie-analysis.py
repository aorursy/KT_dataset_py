# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
# Import the numpy and pandas packages and set options for scrolling the dataframe



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#pd.set_option("display.max_rows", None)

#Read the input csv from the file path in the local computer

movies = pd.read_csv('../input/imdb-movie-data/movie_IMDB.csv') 

movies
#To check the number of Rows and Columns of the data frame 

movies.shape
#To check the list of Columns in the Dataframe 

movies.columns
#to check the max , percentage of data and the stats of the column data 

movies.describe()
#to check the data type data type and the collumn type and the stats of column 

movies.info()
# Write your code for column-wise null count here

movies.isnull().sum(axis=0)
# Write your code for row-wise null count here

movies.isnull().sum(axis=1)

# Write your code for column-wise null percentages here

percentage_miss_columns=movies.isnull().sum() * 100 / len(movies)



percentage_miss_columns
# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 

movies.drop(['color','director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','actor_2_name','cast_total_facebook_likes','actor_3_name','duration','facenumber_in_poster','content_rating','country','movie_imdb_link','aspect_ratio','plot_keywords',], inplace=True, axis=1)

#check columns stats after dropping unwanted columns 

movies.info()
#Check the new shape after dropping the unwanted columns where new rows and columns are obtained 

movies.shape
#Check new data max min and percentage and stastics of the data  based on column altered 

movies.describe()
# Write your code for dropping the rows here

#checking for columns having percentage (greater than 5%) of Null values



(100*(movies.isnull().sum(axis=0)/len(movies.index))>5)

# Drop NAN valus of the columns identified 

movies = movies[~np.isnan(movies['gross'])]

movies = movies[~np.isnan(movies['budget'])]
#Check Percentage of null/ NAN values in the columns after dropping NAN values :

(round(100*(movies.isnull().sum()/len(movies.index)), 2))
# Write your code for filling the NaN values in the 'language' column here

movies.loc[pd.isnull(movies['language']), ['language']]='English'
# Check Percentage of the columns after filling english language for the null places 

(round(100*(movies.isnull().sum()/len(movies.index)), 2))
# Write your code for checking number of retained rows here

len(movies.index)
int((len(movies.index)/5043)*100)
# Write your code for unit conversion here

movies['gross'] = movies['gross']/1000000

movies['budget'] = movies['budget']/1000000



movies

# Write your code for creating the profit column here

movies['profit']=movies['gross']-movies['budget']



movies

# Write your code for sorting the dataframe here

movies.sort_values(by='profit',ascending=False)

# Write code for profit vs budget plot here





a=movies.plot(kind='scatter',x='budget',y='profit',color='blue')

a.set_title('Budget Vs Profit')



plt.show()



 # Write your code to get the top 10 profiting movies here

top10 =movies.sort_values(by='profit',ascending=False)

top10.head(10)
# Write your code for dropping duplicate values here



movies.drop_duplicates(keep=False, inplace=True)

movies
# Write code for repeating subtask 2 here



movies.sort_values(by='profit',ascending=False).head(10)

# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 

# and name that dataframe as 'IMDb_Top_250'



IMDB_Top_250 =movies[movies['num_voted_users']>25000]

IMDB_Top_250.sort_values(by='imdb_score',ascending=False)

IMDB_Top_250['rank'] = np.arange(len(IMDB_Top_250))





IMDB_Top_250.head(250)
 # Write your code to extract top foreign language films from 'IMDb_Top_250' here



Top_Foreign_Lang_Film = IMDB_Top_250[IMDB_Top_250['language']!='English'] 

Top_Foreign_Lang_Film.sort_values(by='imdb_score',ascending=False).head(250)
# Write your code for extracting the top 10 directors here

movies_by_dir=movies.groupby('director_name')

mean_director=movies_by_dir['imdb_score'].mean()

dir_top_ten=mean_director.sort_values(ascending=False).head(10)



dir_top_ten
# Write your code for extracting the first two genres of each movie here

genrenew = movies["genres"].str.split("|", n = 2, expand = True) 

movies['genre_1']= genrenew[0]

movies['genre_2']=genrenew[1]

movies.loc[pd.isnull(movies['genre_2']), ['genre_2']] = movies['genre_1']

movies
# Write your code for grouping the dataframe here



#group by two Genre present in the movies data frame and find the mean of the genres by gross



movies_by_segment =movies.groupby(['genre_1', 'genre_2'])

movies_by_segment=movies_by_segment['gross'].mean()



movies_by_segment
#the Moviesegement grouped is sorted 

PopGenre =movies_by_segment.sort_values(ascending=False)

PopGenre.head(5)
# Write your code for creating three new dataframes here

#dataframe for Meryl Streep

Meryl_Streep = movies[movies.actor_1_name == 'Meryl Streep']



Meryl_Streep
#Data from for Leo_Caprio

Leo_Caprio = movies[movies.actor_1_name == 'Leonardo DiCaprio']

Leo_Caprio
Brad_Pitt = movies[movies.actor_1_name == 'Brad Pitt']

Brad_Pitt 
# Write your code for combining the three dataframes here

actor_group1 = pd.concat([Meryl_Streep, Leo_Caprio], axis = 0)

actor_group2 = pd.concat( [actor_group1, Brad_Pitt], axis = 0)

actor_group2
# Write your code for grouping the combined dataframe here

combinedactor =actor_group2.groupby(['actor_1_name'])

combinedactor



# Write the code for finding the mean of critic reviews and audience reviews here

critic_reviews=combinedactor['num_critic_for_reviews'].mean()

critic_reviews.sort_values(ascending=False)

 

audience_reviews=combinedactor['num_voted_users'].mean()

 

audience_reviews.sort_values(ascending=False)
# Write the code for calculating decade here

movies['decade'] = (movies['title_year'] - movies['title_year']%10).astype(int)

movies
# Write your code for creating the data frame df_by_decade here 

df_by_decade = movies.groupby(["decade"])

df_by_decade=df_by_decade["num_voted_users"].sum().reset_index().sort_values(["decade"])



df_by_decade
# Write your code for plotting number of voted users vs decade



b=df_by_decade.plot(kind='bar',x='decade',y='num_voted_users')

b.set_title('number of voted users vs decade')

plt.yscale('log')

plt.show()