# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
# Import the numpy and pandas packages

import numpy as np
import pandas as pd
 # Write your code for importing the csv file here
movies=pd.read_csv("../input/imdb-movie-data/MovieAssignmentData.csv")
movies.head() # to see the first five rows 
# Write your code for inspection here
movies.columns # to know the column names
movies.shape # to know the number of rows and columns
movies.info()# to know the data types of the column
movies.describe() #to know which column's type data or to find mean average, max and min
# Write your code for column-wise null count here
movies.isnull().sum(axis=1)
# Write your code for row-wise null count here
movies.isnull().sum()
# Write your code for column-wise null percentages here
round(100*(movies.isnull().sum()/len(movies.index)),2) # percentage of null column wise percentage
# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 
movies=movies.drop(["color","director_facebook_likes","actor_1_facebook_likes","actor_2_facebook_likes","actor_3_facebook_likes","actor_2_name","cast_total_facebook_likes","actor_3_name","duration","facenumber_in_poster","content_rating","country","movie_imdb_link","aspect_ratio","plot_keywords"],axis=1)
movies.head() #to inspect the column name
# row size is decreased from 28 to 13
# Write your code for dropping the rows here
movies=movies[~np.isnan(movies["gross"])]
#movies=movies[~np.isnan(movies["content_rating"])]# content rating column dropped #its d ype is object->can we have to convert to float?
movies=movies[~np.isnan(movies["budget"])]
#df=df[~np.isnan(df["aspect_ratio"])] # aspect_ratio column is dropped
round(100*(movies.isnull().sum()/len(movies.index)),2) # to caluclate the percentage 
# Write your code for filling the NaN values in the 'language' column here
movies.loc[pd.isnull(movies["language"]),["language"]]="English"
round(100*(movies.isnull().sum()/len(movies.index)),2)# to see the percentage of column language
# after execution language column changed to 0.00
# Write your code for checking number of retained rows here
movies.shape
round(100*len(movies.index)/5043)
# Write your code for unit conversion here
movies[['budget','gross']].apply(lambda x: round(x/1000000,1))
# Write your code for creating the profit column here
movies['profit']=movies['gross']-movies['budget']
movies.head()
# profit column is created
# Write your code for sorting the dataframe here
movies=movies.sort_values(by=['profit','movie_title'],ascending=False)
movies[0:10]
# Write code for profit vs budget plot here
movies.describe()
import matplotlib.pyplot as plt
import seaborn as sns
# movies.plot(x="budget", y="profit", kind="scatter")
# plt.show()
sns.scatterplot(x="budget",y="profit",data=movies)
plt.show()
# outlier away frome the cluster of values
# Write your code to get the top 10 profiting movies here
top10 = movies.head(10)
top10.movie_title 
# Write your code for dropping duplicate values here
movies.drop_duplicates(keep="first",inplace=True)
# Write code for repeating subtask 2 here
#movies['profit']=movies['gross']-movies['budget']
movies=movies.sort_values(by=['profit','movie_title'],ascending=False)
top10 = movies.head(10)
top10

# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 
# and name that dataframe as 'IMDb_Top_250'
#movies['IMDb_Top_250']=movies['imdb_score']
IMDb_Top_250=movies[movies.num_voted_users>25000].sort_values("imdb_score",ascending=False).head(250)
rank =[]
for i in range(1,251):
    rank.append (i)
IMDb_Top_250['rank']=rank
IMDb_Top_250.head(250)
# Write your code to extract top foreign language films from 'IMDb_Top_250' here
Top_Foreign_Lang_Film = IMDb_Top_250.loc[(IMDb_Top_250.language=='Hindi')]
Top_Foreign_Lang_Film
# Write your code for extracting the top 10 directors here
movies.groupby('director_name')
top10director=movies.groupby('director_name').imdb_score.mean().sort_values(ascending=False)
top10director.head(10)
# Write your code for extracting the first two genres of each movie here
movies['genres']=movies.genres.str.split('|')
movies['genre_1']=movies['genres'].apply(lambda x: x[0])
movies['genre_2']=movies['genres'].apply(lambda x: x[1] if len(x)>1 else x[0])
movies.head()
# Write your code for grouping the dataframe here
movies_by_segment = movies.groupby(['genre_1','genre_2'])
movies_by_segment.gross.mean().sort_values(ascending=False)
# Write your code for getting the 5 most popular combo of genres here
PopGenre = movies_by_segment.gross.mean().sort_values(ascending=False)
PopGenre[0:5]
# Write your code for creating three new dataframes here
# Include all movies in which Meryl_Streep is the lead
Meryl_Streep =movies.loc[movies['actor_1_name']=='Meryl Streep',:]
Meryl_Streep.movie_title
# Include all movies in which Leo_Caprio is the lead
Leo_Caprio =movies.loc[movies['actor_1_name']=='Leonardo DiCaprio',:]
Leo_Caprio.movie_title
# Include all movies in which Brad_Pitt is the lead
Brad_Pitt =movies.loc[movies['actor_1_name']=='Brad Pitt',:]
Brad_Pitt.movie_title
# Write your code for combining the three dataframes here
d1=Meryl_Streep.append(Leo_Caprio)
combined_df=d1.append(Brad_Pitt)
combined_df.head()
# Write your code for grouping the combined dataframe here
df_by_act= combined_df.groupby(['actor_1_name'])
# Write the code for finding the mean of critic reviews and audience reviews here
critic_reviews=df_by_act['num_critic_for_reviews'].mean().sort_values(ascending=False).head(49)
critic_reviews
audience_reviews=df_by_act['num_user_for_reviews'].mean().sort_values(ascending=False).head(49)
audience_reviews
# Write the code for calculating decade here
decade =[]
for i in range(1,3892):
    decade=movies['title_year']//10*10
movies["decade"]=decade.astype(int)
movies.head(2)
# Write your code for creating the data frame df_by_decade here 
df_by_decade=movies.groupby('decade').num_voted_users.sum().sort_values(ascending=True).reset_index()
df_by_decade
# Write your code for plotting number of voted users vs decade
sns.barplot(x="decade",y="num_voted_users",data=movies,ci=68)
plt.show()