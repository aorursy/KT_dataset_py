# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
# Import the numpy and pandas packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Write your code for importing the csv file here

movies =pd.read_csv(r'/kaggle/input/imdb-movie-data/movie_IMDB.csv')

movies
# Write your code for inspection here

print(type(movies.info()))

print(movies.shape)
# Write your code for column-wise null count here

movies.isnull().sum(axis=0)
# Write your code for row-wise null count here

movies.isnull().sum(axis=1)
# Write your code for column-wise null percentages here

round(100*(movies.isnull().sum()/len(movies.index)),2)
# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 

movies=movies.drop(['color','director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','actor_2_name','cast_total_facebook_likes','actor_3_name','duration','facenumber_in_poster','content_rating','country','movie_imdb_link','aspect_ratio','plot_keywords'],axis=1)

round(100*(movies.isnull().sum()/len(movies.index)),2)

# Write your code for dropping the rows here

movies = movies[~np.isnan(movies['gross'])]

movies = movies[~np.isnan(movies['budget'])]

round(100*(movies.isnull().sum()/len(movies.index)),2)
# Write your code for filling the NaN values in the 'language' column here

movies['language'].fillna('English', inplace= True)

round(100*(movies.isnull().sum()/len(movies.index)), 2)
# Write your code for checking number of retained rows here

print('Number of Rowsretained after completing all the tasks above :',movies.shape[0])

print('Percentage of the rows retained after completing all the tasks above',round(100*(len(movies.index)/5043),2))
# Write your code for unit conversion here

movies['budget']=movies['budget']/1000000

movies['gross']=movies['gross']/1000000



movies
# Write your code for creating the profit column here

movies['profit']=movies['gross'] - movies['budget']

movies

# Write your code for sorting the dataframe here

movies.sort_values(by='profit',ascending=True)
# Write code for profit vs budget plot here

#Plot profit (y-axis) vs budget (x- axis) and observe the outliers using the appropriate chart type

ax = movies.plot.scatter(x="budget", y="profit", color='r',figsize=(16,8))

ax.set_yscale('log')

ax.set_xscale('log')

plt.xlabel("Budget in million $")

plt.ylabel("Profit in million $")

plt.show()
# Write your code to get the top 10 profiting movies here

top10=movies.sort_values(by=['profit'],ascending=False).head(10)

top10
# Write your code for dropping duplicate values here

movies.drop_duplicates()
# Write code for repeating subtask 2 here

top10=movies.sort_values(by=['profit'],ascending=False).head(10)

top10
# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 

# and name that dataframe as 'IMDb_Top_250'

IMDb_Top_250 = movies

IMDb_Top_250.sort_values(["imdb_score"], axis=0,ascending=False, inplace=True) 

IMDb_Top_250=IMDb_Top_250[IMDb_Top_250['num_voted_users']>25000]

IMDb_Top_250=IMDb_Top_250.head(250)

IMDb_Top_250['Rank']=list(np.arange(1,251))

IMDb_Top_250
 # Write your code to extract top foreign language films from 'IMDb_Top_250' here

Top_Foreign_Lang_Film =IMDb_Top_250[IMDb_Top_250['language']!='English']

Top_Foreign_Lang_Film
# Write your code for extracting the top 10 directors here

subtask3_5=movies.groupby('director_name')

top10director=pd.DataFrame(subtask3_5['imdb_score'].mean().sort_values(ascending=False))

top10director=top10director.head(10)

top10director=top10director.sort_values(['imdb_score','director_name'],ascending=(False,True))

top10director
# Write your code for extracting the first two genres of each movie here

moviegenres=movies['genres'].apply(lambda x: pd.Series(x.split('|')))

movies['genre_1']=moviegenres[0]

movies['genre_2']=moviegenres[1]

movies.loc[pd.isnull(movies['genre_2']), ['genre_2']] = movies['genre_1']

movies=movies.drop(['genres'], axis = 1) 

movies
# Write your code for grouping the dataframe here

movies_by_segment =movies.groupby(['genre_1','genre_2'])

movies
# Write your code for getting the 5 most popular combo of genres here

PopGenre = movies_by_segment['gross'].mean().sort_values(ascending=False).head(5)

PopGenre
# Write your code for creating three new dataframes here



# Include all movies in which Meryl_Streep is the lead

Meryl_Streep=movies.loc[(movies.actor_1_name=='Meryl Streep'),:]

Meryl_Streep
# Include all movies in which Leo_Caprio is the lead

Leo_Caprio=movies.loc[(movies.actor_1_name=='Leonardo DiCaprio'),:]

Leo_Caprio
# Include all movies in which Brad_Pitt is the lead

Brad_Pitt=movies.loc[(movies.actor_1_name=='Brad Pitt'),:]

Brad_Pitt 
# Write your code for combining the three dataframes here

Combined=Meryl_Streep.append(Leo_Caprio).append(Brad_Pitt)

Combined
# Write your code for grouping the combined dataframe here

actor_name=Combined.groupby('actor_1_name')
# Write the code for finding the mean of critic reviews and audience reviews here

critic_reviews=actor_name['num_critic_for_reviews'].mean().sort_values(ascending=False)

print(critic_reviews)

audience_reviews=actor_name['num_user_for_reviews'].mean().sort_values(ascending=False)

print(audience_reviews)
# Write the code for calculating decade here

movies['decade']=movies['title_year'].apply(lambda x: x//10 * 10 ).astype(np.int64)

movies['decade']=movies['decade'].astype(str)+'s'

movies=movies.sort_values(['decade'])

movies
# Write your code for creating the data frame df_by_decade here 

df_by_decade=movies.groupby(['decade'])

df_by_decade['num_voted_users'].sum()

df_by_decade=pd.DataFrame(df_by_decade['num_voted_users'].sum())

df_by_decade
# Write your code for plotting number of voted users vs decade



ax= df_by_decade.plot.bar(figsize=(16,8));

ax.set_yscale('log')

plt.xlabel("Decade")

plt.ylabel("Number of Voted Users")

plt.title("Number of Voted Users vs Decade")

plt.show()

 

            