# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Import the numpy and pandas packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline  
sns.set_style("whitegrid")

 # Write your code for importing the csv file here
movies = pd.read_csv("data.csv")
pd.options.display.float_format = '{:.2f}'.format
movies
# Write your code for inspection here
print(movies.columns)
print(movies.shape)
print(movies.dtypes)
movies.info()
movies.describe()
# Write your code for column-wise null count here
movies.isnull().sum(axis=0)
# Write your code for row-wise null count here
movies.isnull().sum(axis=1)
# Write your code for column-wise null percentages here
percentage=round(100*(movies.isnull().sum(axis=0)/len(movies.index)),2)
print(percentage)
# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 
movies=movies.drop(['color','director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','actor_2_name','cast_total_facebook_likes','actor_3_name','duration','facenumber_in_poster','content_rating','country','movie_imdb_link','aspect_ratio','plot_keywords'],axis=1)
#print(movies.columns)
#percentage=round(100*(movies.isnull().sum(axis=0)/len(movies.index)),2)
#print(percentage)                   
# Write your code for dropping the rows here
#gross,budget > 5%
movies = movies[~np.isnan(movies['gross'])]
movies = movies[~np.isnan(movies['budget'])]
#percentage=round(100*(movies.isnull().sum(axis=0)/len(movies.index)),2)
#print(percentage)
#print(movies.shape)
# Write your code for filling the NaN values in the 'language' column here
movies['language'].fillna('English', inplace = True) 
#movies.loc[pd.isnull(movies['language'])] = 'English'

# Write your code for checking number of retained rows here


retained_rows=len(movies.index)
print(retained_rows)

#intially shape =(5043, 28)
#percentage_retained_rows= ((3891)/5043)*100
#print(percentage_retainedrows)
# Write your code for unit conversion here
movies['budget']= movies['budget'].map(lambda x: x/1000000)
movies['gross']= movies['gross'].map(lambda x: x/1000000)

# Write your code for creating the profit column here
movies['profit']= movies['gross']-movies['budget']
# Write your code for sorting the dataframe here
movies=movies.sort_values(by=['profit'], ascending=False)
print(movies)
# Write code for profit vs budget plot here

#movies['profit'].describe()
#movies['budget'].describe()
g = sns.jointplot(x='budget', y='profit', data=movies)
# Write your code to get the top 10 profiting movies here
top10=movies.iloc[0:10]
print(top10)
# Write your code for dropping duplicate values here
movies = movies.drop_duplicates(subset=['movie_title','language'], keep='first')
movies['profit']= movies['gross']-movies['budget']
movies=movies.sort_values(by=['profit'], ascending=False)
# Write code for repeating subtask 2 here
g = sns.jointplot(x='budget', y='profit', data=movies)
top10=movies.head(10)
print(top10)

# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 
# and name that dataframe as 'IMDb_Top_250'
IMDb_Top_250=movies[movies['num_voted_users'] > 25000].sort_values(by=['imdb_score'], ascending=False).iloc[0:250]
IMDb_Top_250['Rank'] = range(1, 1+len(IMDb_Top_250))
print(IMDb_Top_250)
Top_Foreign_Lang_Film = IMDb_Top_250[IMDb_Top_250['language']!= 'English']
print(Top_Foreign_Lang_Film)
# Write your code for extracting the top 10 directors here
top10director=movies.groupby('director_name',as_index=False)["imdb_score"].mean().sort_values(by=["imdb_score",'director_name'],ascending=[False,True]).iloc[0:10]
print(top10director)

# Write your code for extracting the first two genres of each movie here

col2 = movies['genres'].apply(lambda x: pd.Series(x.split('|'))).rename(columns={0:'genre1', 1:'genre2',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7'}).ffill(axis=1)
col2=col2.drop(['2','3','4','5','6','7'], axis=1)
movies.insert(3,'genre_1',col2.genre1)
movies.insert(4,'genre_2',col2.genre2)
movies=movies.drop('genres', axis=1)
print(movies)

# Write your code for grouping the dataframe here
PopGenre=movies.groupby(['genre_1','genre_2'],as_index=False)["gross"].mean()
print(PopGenre)
# Write your code for getting the 5 most popular combo of genres here
PopGenre.sort_values(by=['gross'],ascending=False).iloc[0:5]

# Write your code for creating three new dataframes here
#actors=movies['actor_1_name'].tolist()
Meryl_Streep = movies.loc[movies['actor_1_name'] == "Meryl Streep",:]
 # Include all movies in which Leo_Caprio is the lead
Leo_Caprio = movies.loc[movies['actor_1_name'] == "Leonardo DiCaprio",:]
# Include all movies in which Brad_Pitt is the lead
Brad_Pitt =  movies.loc[movies['actor_1_name'] == "Brad Pitt",:]
# Write your code for combining the three dataframes here
Combined=pd.concat([Meryl_Streep, Leo_Caprio,Brad_Pitt], axis = 0)
# Write your code for grouping the combined dataframe here
Combined.groupby('actor_1_name').apply(lambda x:x)
# Write the code for finding the mean of critic reviews and audience reviews here 
# mean of the num_critic_for_reviews and num_users_for_review and identify the actors which have the highest mean.

Combined.groupby('actor_1_name').agg({"num_critic_for_reviews":"mean","num_user_for_reviews":"mean"})

# Write the code for calculating decade here

df_by_decade=movies.copy(deep=True)
df_by_decade['decade']=df_by_decade['title_year'].apply(lambda x:10*(int(x/10)))

# Write your code for creating the data frame df_by_decade here 
df_by_decade=df_by_decade.groupby('decade',as_index=False)['num_voted_users'].sum().sort_values(by="decade")
# Write your code for plotting number of voted users vs decade
ax=sns.barplot(x='decade',y='num_voted_users',data=df_by_decade)
plt.yscale('log')
plt.ylabel("num_voted_users")
plt.xlabel("decade")
plt.show()

