# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
# Import the numpy and pandas packages



import numpy as np

import pandas as pd
# Write your code for importing the csv file here

movies = pd.read_csv("../input/imdb-movie-dataset/MovieAssignmentData (3).csv")

movies
# Write your code for inspection here

print(movies.ndim)

print(movies.info())

print(movies.dtypes)

print(movies.columns)

print(movies.shape)

movies.describe



# Write your code for column-wise null count here

movies.isnull().sum()
# Write your code for row-wise null count here

movies.isnull().sum(axis=1)
# Write your code for column-wise null percentages here

round(100*(movies.isnull().sum()/len(movies.index)),2)
# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 

# Dropping "color" column

movies = movies.drop('color', axis=1)

print(movies.info())

print(movies.dtypes)

print(movies.columns)

print(movies.shape)
# Dropping "director_facebook_likes" column

movies = movies.drop('director_facebook_likes', axis=1)

print(movies.info())

print(movies.dtypes)

print(movies.columns)

print(movies.shape)
# Dropping "actor_1_facebook_likes" column

movies = movies.drop('actor_1_facebook_likes', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "actor_2_facebook_likes" column

movies = movies.drop('actor_2_facebook_likes', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "actor_3_facebook_likes" column

movies = movies.drop('actor_3_facebook_likes', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "actor_2_name" column

movies = movies.drop('actor_2_name', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "cast_total_facebook_likes" column

movies = movies.drop('cast_total_facebook_likes', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "actor_3_name" column

movies = movies.drop('actor_3_name', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "duration" column

movies = movies.drop('duration', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "facenumber_in_poster" column

movies = movies.drop('facenumber_in_poster', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "content_rating" column

movies = movies.drop('content_rating', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "country" column

movies = movies.drop('country', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "movie_imdb_link" column

movies = movies.drop('movie_imdb_link', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "aspect_ratio" column

movies = movies.drop('aspect_ratio', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
# Dropping "plot_keywords" column

movies = movies.drop('plot_keywords', axis=1)

print(movies.info())

print(movies.columns)

print(movies.shape)
#Inspecting to find out the percentage of null values in the columns:

round(100*(movies.isnull().sum()/len(movies.index)), 2)
# Write your code for dropping the rows here

#Dropping the rows for which the "budget" column has NaN 

movies.dropna(subset = ["budget"], inplace=True)

round(100*(movies.isnull().sum()/len(movies.index)), 2)
# Write your code for dropping the rows here

#Dropping the rows for which the "gross" column has NaN 

movies.dropna(subset = ["gross"], inplace=True)

round(100*(movies.isnull().sum()/len(movies.index)), 2)
# Write your code for filling the NaN values in the 'language' column here

movies['language'] = movies['language'].fillna('English')
# Inspecting the percent of Nan in columns after filling English in above step:

round(100*(movies.isnull().sum()/len(movies.index)), 2)
# Write your code for checking number of retained rows here

(len(movies.index)/5043)*100
# Write your code for unit conversion here

movies["gross"] = movies["gross"]/1000000

movies
# Write your code for unit conversion here

movies["budget"] = movies["budget"]/1000000

movies
# Write your code for creating the profit column here

movies["profit"]=movies["gross"]-movies["budget"]

movies
# Write your code for sorting the dataframe here

movies.sort_values(by=['profit'])

# Write code for profit vs budget plot here

import matplotlib .pyplot as plt

plt.scatter(movies['budget'],movies['profit'],color='g',marker='o')

plt.title("Budget VS Profit")

plt.xlabel('budget')

plt.ylabel('profit')

plt.show()
# Write your code to get the top 10 profiting movies here

top10 = movies.sort_values(by=['profit'],ascending=False)

top10 = top10.head(10)

top10

print("TOP 10 MOVIE TITLES: ")

top10[['movie_title','language']]
# Write your code for dropping duplicate values here

#movies=

movies = movies.drop_duplicates(subset=['movie_title','language'],keep='first')

movies
# Write code for repeating subtask 2 here



# Step1:Create a new column called profit which contains the difference of the two columns: gross and budget.



# Write your code for creating the profit column here



movies["profit"]=movies["gross"]-movies["budget"]

movies
# Write code for repeating subtask 2 here



# Step-2 Sort the dataframe using the profit column as reference.



# Write your code for sorting the dataframe here



movies.sort_values(by=['profit'])

# Write code for repeating subtask 2 here



#Step-3 Plot profit (y-axis) vs budget (x- axis) and observe the outliers using the appropriate chart type.



# Write code for profit vs budget plot here



import matplotlib .pyplot as plt

plt.scatter(movies['budget'],movies['profit'],color='g',marker='o')

plt.title("Budget VS Profit")

plt.xlabel('budget')

plt.ylabel('profit')

plt.show()
# Write code for repeating subtask 2 here



# Step 4 : Extract the top ten profiting movies in descending order and store them in a new dataframe - top10



# Write your code to get the top 10 profiting movies here



top10 = movies.sort_values(by=['profit'],ascending=False)

top10 = top10.head(10)

top10

print("TOP 10 MOVIE TITLES: ")

top10[['movie_title','language','director_name']]
# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 

# and name that dataframe as 'IMDb_Top_250'



##Step 1 )Creating a new dataframe as 'IMDb_Top_250' by applying the filter : "num_voted_users">25000 



IMDb_Top_250=movies[movies["num_voted_users"]>25000]



#Selecting just 4 columns of the dataframe inorder to see the changes we are making to the dataframe 

# clearly 



IMDb_Top_250[["movie_title","imdb_score","num_voted_users"]]
# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 

# and name that dataframe as 'IMDb_Top_250'



##Step 2)Applying filter:Sorting the dataframe IMDb_Top_250 with the highest IMDb Rating(corresponding to the column: imdb_score).





IMDb_Top_250=IMDb_Top_250.sort_values(by=['imdb_score'],ascending=False)



#Selecting just 4 columns of the dataframe inorder to see the changes we are making to the dataframe easily 

# and efficiently

IMDb_Top_250[["movie_title","imdb_score","num_voted_users"]]
# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 

# and name that dataframe as 'IMDb_Top_250'



##Step 3 ) Storing the top 250 movies in the Dataframe : IMDb_Top_250



IMDb_Top_250=IMDb_Top_250.head(250)



#Selecting just 4 columns of the dataframe inorder to see the changes we are making to the dataframe easily 

# and efficiently

IMDb_Top_250[["movie_title","imdb_score","num_voted_users"]]
# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 

# and name that dataframe as 'IMDb_Top_250'



## Step 4)Adding a "Rank" column containing the values 1 to 250 indicating the ranks of the corresponding films. 



IMDb_Top_250["Rank"]=np.arange(1,len(IMDb_Top_250)+1)



#Selecting just 4 columns of the dataframe inorder to see the changes we are making to the dataframe easily 

# and efficiently

IMDb_Top_250[["movie_title","imdb_score","num_voted_users","Rank"]]

## Having a  whole look at the final and complete dataframe:  IMDb_Top_250

IMDb_Top_250
# Write your code to extract top foreign language films from 'IMDb_Top_250' here



Top_Foreign_Lang_Film =IMDb_Top_250[IMDb_Top_250["language"]!="English"]



#Selecting just 3 columns of the dataframe inorder to see the changes we are making to the dataframe easily 

# and efficiently

Top_Foreign_Lang_Film[["movie_title","language","Rank"]]
## Having a  whole look at the final and complete dataframe:  Top_Foreign_Lang_Film

Top_Foreign_Lang_Film
# Write your code for extracting the top 10 directors here

top10director=movies.sort_values("director_name")

top10director=top10director.groupby('director_name').imdb_score.mean()



top10director=top10director.nlargest(10)

top10director=pd.DataFrame(top10director)

top10director
# Write your code for extracting the first two genres of each movie here



## Step 1) Splitting the string in the genres column 2 times on the separator='|' and storing the 

# #       respective values into the three columns : "genre_1","genre_2","to be dropped".

#         This 3rd column : "to be dropped"  will be dropped in the next step because we do not need

#         these values ie., we only need to extract 1st two genres in the columns "genre_1" & "genre_2".



movies[["genre_1","genre_2","to be dropped"]]=movies.genres.str.split('|',2,expand=True)

movies[['movie_title','genres',"genre_1","genre_2","to be dropped"]]
# Write your code for extracting the first two genres of each movie here



# Step 2) Dropping the column:"to be dropped"



movies=movies.drop("to be dropped",axis=1)

movies
# Write your code for extracting the first two genres of each movie here



# Step 3) filling the Null value in the 'genre_2' column with the value in the 'genre_1' column.



movies.genre_2.fillna(movies.genre_1,inplace=True)

movies

# Write your code for grouping the dataframe here



# Also arranging in descending order of mean gross to find top popular movies genres



movies_by_segment = movies.groupby(["genre_1","genre_2"]).gross.mean().sort_values(ascending=False)

movies_by_segment=pd.DataFrame(movies_by_segment)

movies_by_segment
# Write your code for getting the 5 most popular combo of genres here



PopGenre = movies_by_segment.head(5)

PopGenre
# Write your code for creating three new dataframes here



# Include all movies in which Meryl_Streep is the lead



Meryl_Streep =movies.loc[movies['actor_1_name']=="Meryl Streep"] 

Meryl_Streep
# Include all movies in which Leo_Caprio is the lead

Leo_Caprio = movies.loc[movies['actor_1_name']=="Leonardo DiCaprio"]

Leo_Caprio
# Include all movies in which Brad_Pitt is the lead



Brad_Pitt = movies.loc[movies['actor_1_name']=='Brad Pitt']

Brad_Pitt
# Write your code for combining the three dataframes here

Combined=Meryl_Streep.append(Leo_Caprio)

Combined=Combined.append(Brad_Pitt)

Combined
# Write your code for grouping the combined dataframe here



Combined_group=Combined.groupby("actor_1_name")

Combined_group.size()
# Write the code for finding the mean of critic reviews and audience reviews here



Combined_1=pd.DataFrame(Combined.groupby("actor_1_name")[['num_critic_for_reviews','num_user_for_reviews']].mean())

Combined_1=Combined_1.sort_values(by=['num_critic_for_reviews','num_user_for_reviews'],ascending=False)

Combined_1
# Write the code for calculating decade here



movies['decade']=((movies.title_year//10)*10)



movies
# Write your code for creating the data frame df_by_decade here 



df_by_decade=movies.groupby('decade')['num_voted_users'].sum()

df_by_decade=pd.DataFrame(df_by_decade)

df_by_decade

df_by_decade.columns
## Converting the decade column datatype into integer data type.



df_by_decade.index=df_by_decade.index.astype(int)

df_by_decade



# Write your code for plotting number of voted users vs decade



import matplotlib .pyplot as plt

fig= plt.figure(figsize=(12,8))



axes= fig.add_axes([0.1,0.1,0.8,0.8])

axes.bar(df_by_decade.index,df_by_decade['num_voted_users'],color='Purple',label='bar plot')

plt.title("number of voted users vs decade")

plt.xlabel('decade')

plt.xlim([1910,2020])

plt.ylabel('number of voted users')

plt.legend()

plt.show()
### using the yscale as logarithmic for a better view of the graph.



import matplotlib .pyplot as plt

fig= plt.figure(figsize=(12,8))



axes= fig.add_axes([0.1,0.1,0.8,0.8])

axes.bar(df_by_decade.index,df_by_decade['num_voted_users'],color='Purple',label='bar plot')

plt.title("number of voted users vs decade")

plt.xlabel('decade')

plt.xlim([1910,2020])

plt.ylabel('number of voted users')

plt.yscale('log')

plt.legend()

plt.show()