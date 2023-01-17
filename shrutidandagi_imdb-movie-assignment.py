# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
# Import the numpy and pandas packages

import numpy as np
import pandas as pd

#Write Code for importing the csv file here
movies= pd.read_csv('../input/imdb-movie-dataset/MovieAssignmentData.csv')
movies
# Write your Code for inspection here  
#Column names in the dataframe
movies.columns
#The number of rows and columns
movies.shape

#summary of all the numeric columns in the dataset
movies.describe()    #movies.describe(include="all")
#Datatypes of each column
movies.info()
# Write your Code for column-wise null count 
movies.isnull().sum()
# Write your Code for row-wise null count 
movies.isnull().sum(axis=1)
#Write your code for Column-wise null percentages 
round(100*(movies.isnull().sum()/len(movies.index)),2)
#Write your Code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 
movies=movies.drop(['color','director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','actor_2_name','cast_total_facebook_likes','actor_3_name','duration','facenumber_in_poster','content_rating','country','movie_imdb_link','aspect_ratio','plot_keywords'],axis=1)
round(100*(movies.isnull().sum()/len(movies.index)),2)        #axis=1 for columns

# Write your Code for dropping the rows here
movies=movies[~np.isnan(movies['gross'])]
movies=movies[~np.isnan(movies['budget'])]

round(100*(movies.isnull().sum()/len(movies.index)),2)

#Write your Code for filling the NaN values in the 'language' column here
movies.loc[pd.isnull(movies['language']),['language']]="English"   #Replacing missing values in language column with English
round(100*(movies.isnull().sum()/len(movies.index)))
# Write your Code for checking number of retained rows  here
movies.shape
100*len(movies.index)/5043       #Dividing movies.index with total number of rows

# Write your code for unit conversion here
movies['budget']=movies['budget'].apply(lambda x: round(x/1000000,1))   
movies['gross']=movies['gross'].apply(lambda x: round(x/1000000,1))
movies.head()
# Write your code for creating the profit column here
movies['profit']=movies['gross']-movies['budget']
movies.head()
# Write your code for sorting the dataframe here
movie=movies.sort_values(by=['profit'],ascending=False)
movie.head()
# Write code for profit vs budget plot here
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(num=None, figsize=(12,7), dpi=80)   
movie=movie[movie.profit>100]
plt.scatter(movie['budget'], movie['profit'],marker ="H",edgecolors = 'b',facecolor='lightblue') #Marker= hexagon, Edgecolor=Blue, fillcolor=lightblue
plt.xlabel("Budget") #Labelling x
plt.ylabel("Profit") #Labelling y
plt.title("Budget-Profit Chart")
plt.grid(color='black', linestyle='-', linewidth=0.25, alpha=0.5)  ##adds major gridlines
plt.show()
# Write your code to get the top 10 profiting movies here
top10=movie[['movie_title']][0:10]  # CreatingNew Dataframe
top10
# Write your code for dropping duplicate values here
movies=movies.drop_duplicates()
movies
# Write code for repeating subtask 2 here
movies['profit']=movies['gross']-movies['budget']
movie=movies.sort_values(by=['profit'],ascending=False)
top10=movie[['director_name','movie_title']] #Creating a Dataframe
top10.head(10)
#type(top10)
# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 
# and name that dataframe as 'IMDb_Top_250'
IMDb_Top_250=movies[['imdb_score','num_voted_users','movie_title','language']]

IMDb_sort= IMDb_Top_250.sort_values(by=['imdb_score'],ascending=False)
IMDb_Top_250=IMDb_sort[IMDb_Top_250.num_voted_users>25000]
IMDb_Top_250.head(250)

#Rank column containing the values 1 to 250 indicating the ranks of the corresponding films.
IMDb_Top_250["Rank"]=IMDb_Top_250['movie_title'].rank()
IMDb_Top_250['Rank']=IMDb_Top_250['Rank'].sort_values(ascending=True).values
IMDb_Top_250.head()
#Setting the Rank column as the index
IMDb_Top_250=IMDb_Top_250.set_index('Rank')
IMDb_Top_250.head(250)
# Write your code to extract top foreign language films from 'IMDb_Top_250' here
Top_Foreign_Lang_Film = IMDb_Top_250[(IMDb_Top_250.language=='Hindi') ] #Extracting foreign langauge movies Hindi being Indian language
Top_Foreign_Lang_Film[0:250]
# Write your code for extracting the top 10 directors here
mov=movies.groupby('director_name')

top10director=pd.DataFrame(mov['imdb_score'].mean().sort_values(ascending=False))  #Convert to Dataframe
top10director=top10director.head(10)

top10director=top10director.sort_values(['imdb_score','director_name'],ascending=(False,True))
top10director
# Write your code for extracting the first two genres of each movie here
movies['genres']=movies.genres.str.split('|')
movies['genre_1']=movies['genres'].apply(lambda x: x[0])  #Extracting one Genre
movies['genre_2']=movies['genres'].apply(lambda x: x[1] if len(x)>1 else x[0])   #Extracting first and second
movies.head()
# Write your code for grouping the dataframe here
movies_by_segment = movies.groupby(['genre_1','genre_2'])
movies_by_segment
   
# Write your code for getting the 5 most popular combo of genres here
PropGenre=pd.DataFrame(movies_by_segment.gross.mean().sort_values(ascending=False) )  #Creating a dataframe
PropGenre[0:5]
# Write your code for creating three new dataframes here
Meryl_Streep=movies[['actor_1_name','movie_title','num_critic_for_reviews','num_user_for_reviews']]
Leo_Caprio=movies[['actor_1_name','movie_title','num_critic_for_reviews','num_user_for_reviews']]
Brad_Pitt=movies[['actor_1_name','movie_title','num_critic_for_reviews','num_user_for_reviews']]

# Include all movies in which Meryl_Streep is the lead
Meryl_Streep=Meryl_Streep.loc[Meryl_Streep['actor_1_name']=='Meryl Streep',:]
Meryl_Streep.head()
# Include all movies in which Leo_Caprio is the lead
Leo_Caprio=Leo_Caprio.loc[Leo_Caprio['actor_1_name']=='Leonardo DiCaprio',:]
Leo_Caprio.head()
# Include all movies in which Brad_Pitt is the lead
Brad_Pitt=Brad_Pitt.loc[Brad_Pitt['actor_1_name']=='Brad Pitt',:]
Brad_Pitt.head()
# Write your code for combining the three dataframes here
Combined=Meryl_Streep.append(Leo_Caprio).append(Brad_Pitt)

Combined
# Write your code for grouping the combined dataframe here
Actor_name=Combined.groupby('actor_1_name')
Actor_name
# Write the code for finding the mean of critic reviews and audience reviews here
# mean of critic reviews
Critic_reviews=Actor_name['num_critic_for_reviews'].mean().sort_values(ascending=False)
Critic_reviews

# mean of audience reviews
Audience_reviews=Actor_name['num_user_for_reviews'].mean().sort_values(ascending=False)
Audience_reviews.head()
#Leonardo has more Critic Reviews and Audience reveiws.
# Write the code for calculating decade here
movies['decade']=movies['title_year'].apply(lambda x: (x//10) *10).astype(np.int64)  #astype(np.int64) to remove .0
movies['decade']=movies['decade'].astype(str)+'s'    #astype(str)+'s' to add s to decade
movies=movies.sort_values(['decade'])
movies
# Write your code for creating the data frame df_by_decade
df_by_decade=movies.groupby('decade')
df_by_decade['num_voted_users'].sum()
#Convert to Dafaframe
df_by_decade=pd.DataFrame(df_by_decade['num_voted_users'].sum())
df_by_decade
# Write your code for plotting number of voted users vs decade
import matplotlib.pyplot as plt

df_by_decade.plot.bar(figsize=(15,8),width=0.8,hatch="//",edgecolor='k')   #Figure size, width of bars, Pattern, edgecolor
plt.xlabel("Decade")
plt.ylabel("Voted Users")
plt.title("Voted users over Decades")
plt.yscale('log')  #Changing y scale
plt.show()
