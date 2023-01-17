# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
# Import the numpy and pandas packages

import numpy as np
import pandas as pd
movies = pd.read_csv('/Users/karthick/Downloads/Movie+Assignment+Data.csv')
movies
# Write your code for inspection here
movies.shape
movies.info()

# Write your code for column-wise null count here
movies.isnull().sum()
# Write your code for row-wise null count here
for i in range(len(movies.index)) :
        print("Number of Null in row ", i , " : " ,  movies.iloc[i].isnull().sum())
# Write your code for column-wise null percentages here
round(movies.isnull().mean()*100,2) 
# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 
movies.drop(['color','director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','actor_2_name','cast_total_facebook_likes','actor_3_name','duration','facenumber_in_poster','content_rating','country','movie_imdb_link','aspect_ratio','plot_keywords'],inplace=True, axis=1, errors='ignore')
movies

# Write your code for dropping the rows here
round(movies.isnull().mean()*100,2) 
movies.dropna(subset=['gross'], inplace=True)

movies.dropna(subset=['budget'], inplace=True)
round(movies.isnull().mean()*100,2) 
# Write your code for filling the NaN values in the 'language' column here
movies['language'].replace( np.nan,'English', inplace=True)
# Write your code for checking number of retained rows here
round(movies.isnull().mean()*100,2)
movies.shape
a=3891/5043*100
a
# Write your code for unit conversion here
movies['budget'] = movies['budget']/1000000
movies['gross'] = movies['gross']/1000000
movies
# Write your code for creating the profit column here
movies['profit'] = movies['gross'] - movies['budget']
movies
# Write your code for sorting the dataframe here
movies.sort_values(by=['profit'], inplace=True, ascending=False)
movies
# Write code for profit vs budget plot here
movies.plot(x='budget',y='profit')
top10 = movies.head(10)
top10
# Write your code for dropping duplicate values here
movies.drop_duplicates(keep=False, inplace=True)
movies

# Write code for repeating subtask 2 here
movies.sort_values(by=['profit'], inplace=True, ascending=False)
top10 = movies.head(10)
top10

# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 
# and name that dataframe as 'IMDb_Top_250'

movies.sort_values(by=['imdb_score'], inplace=True, ascending=False)

Temp = movies[movies['num_voted_users'] > 25000]
IMDb_Top_250 = Temp.head(250)
#IMDb_Top_250['Rank'] = range(1,1+len(IMDb_Top_250)
IMDb_Top_250.insert(0, 'Rank', range(1, 1 + len(IMDb_Top_250)))
IMDb_Top_250
Top_Foreign_Lang_Film = IMDb_Top_250[IMDb_Top_250['language'] != 'English']
Top_Foreign_Lang_Film
# Write your code for extracting the top 10 directors here
df_mean = movies.groupby(['director_name'])[['imdb_score']].mean()
df_mean.sort_values(by=['imdb_score','director_name'], inplace=True, ascending=[False,True])
top10director = df_mean.head(10)
top10director
# Write your code for extracting the first two genres of each movie here
# split_genre=pd.DataFrame(columns=['genre_1','genre_2','remainder'])
movies[['genre_1','genre_2','remainder']]= movies['genres'].str.split("|",expand=True,n=2)
#split_genre.genre_2 = split_genre.genre_1.where(split_genre.genre_2 == 'None', split_genre.genre_2)
movies['genre_2'].replace( np.nan,movies['genre_1'], inplace=True)
movies.drop(['remainder'],inplace=True, axis=1, errors='ignore')

movies
Group_by_genre=movies.groupby(['genre_1','genre_2'])
Group_by_genre
PopGenre = movies.groupby(['genre_1','genre_2'], sort=False) ['gross'].mean()
PopGenre.sort_values(ascending=False).head(1)
      
#print (df_by_spec_count.sort_values(by='count',ascending=False).head())
# Write your code for creating three new dataframes here

Meryl_Streep = movies[movies['actor_1_name'] == 'Meryl Streep']
Meryl_Streep
Leo_Caprio = movies[movies['actor_1_name'] == 'Leonardo DiCaprio']
Leo_Caprio
Brad_Pitt = movies[movies['actor_1_name'] == 'Brad Pitt']
Brad_Pitt
# Write your code for combining the three dataframes here
temp=[Meryl_Streep,Leo_Caprio,Brad_Pitt]
Combined = pd.concat(temp)
Combined

# Write your code for grouping the combined dataframe here
Group_by_actor=Combined.groupby(['actor_1_name'])
Group_by_actor
# Write the code for finding the mean of critic reviews and audience reviews here
Combined.groupby(['actor_1_name']) ['num_critic_for_reviews','num_user_for_reviews'].mean()
#Combined.groupby(['actor_1_name']) ['num_user_for_reviews'].mean()

movies['decade'] = (movies['title_year']//10)*10
movies.sort_values(by=['decade'], inplace=True, ascending=False)
movies
# Write your code for creating the data frame df_by_decade here 
df_by_decade=movies.groupby(['decade'])['num_voted_users'].sum()
df_by_decade
# Write your code for plotting number of voted users vs decade
df_by_decade.plot.bar(x='decade',y='num_voted_users')
