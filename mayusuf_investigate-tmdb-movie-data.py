# Necessay Libraries are included

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# load TMdb Movie data in data frame

df_tmdb = pd.read_csv("/kaggle/input/tmdb-movies.csv")

df_tmdb.head(5)
# Find shape(row, col) of the data set

df_tmdb.shape
# Overall information of the data type and presence and missing of value of column 

df_tmdb.info()
df_tmdb.isnull().sum()
# Summary of Statistical presence of data over the columns that have numerial data type. 

df_tmdb.describe()
# The output(in above) of functions shows that there are few columns of missing rows or values. Now fill null with

# emplty string 

df_tmdb.genres.fillna('', inplace=True)

df_tmdb.cast.fillna('', inplace=True)
df_tmdb.isnull().sum()
# Split the given sting by pipline character 

def explode_string(data):

    return data.split('|')
df_tmdb.genres = df_tmdb.genres.apply(explode_string)

df_tmdb.cast = df_tmdb.cast.apply(explode_string)
popu= 0.7

def movies_based_on_popularity(movies):

    """

    filter function to threshold off popularity column 

    """

    return movies[movies['popularity'] > popu]
df_popular = df_tmdb.groupby('release_year').apply(movies_based_on_popularity)

df_popular.describe()
def movies_genres_year(movies):

    """

        This function takes a Data and returns a dictionary having

        release_year and genre(array) as key and count of genre as value.

    """

    movies_in_year_by_genre = {} # declare of dictionary

    for (release_year,pos), genres in movies.items():

        for genre in genres:

            if release_year in movies_in_year_by_genre:  #check year is exist

                if genre in movies_in_year_by_genre[release_year]: # check  genre is exist

                    movies_in_year_by_genre[release_year][genre] += 1 # increment genre if it is exist

                else:

                    movies_in_year_by_genre[release_year][genre] = 1  # new genre entry 

            else:

                movies_in_year_by_genre[release_year] = {} # declare dictionary within dictionary

                movies_in_year_by_genre[release_year][genre] = 1 # new genre and year entry

                

    return movies_in_year_by_genre
popular_movies = movies_genres_year(df_popular.genres)

#popular_movies
# draw graph of every five years

for year,genre_dic in popular_movies.items():

    if year%5 == 0:

        pd.DataFrame(popular_movies[year], index=[year]).plot(kind='bar', title="Popular Movie by Genre", figsize=(20, 6))
higest_earned_movies = df_tmdb[df_tmdb['revenue_adj']>=1600000000].sort_values(by='revenue_adj',ascending=False)

higest_earned_movies.shape
pd.DataFrame(list(higest_earned_movies.popularity), index=list(higest_earned_movies.original_title)).plot(kind='bar',

title='Popularity of Highest earned movies; Q3 = 0.713817',legend= False,figsize=(20, 10))

plt.show()
pd.DataFrame(list(higest_earned_movies.vote_count), index=list(higest_earned_movies.original_title)).plot(kind='bar',

title='Vote Count of Highest earned movies; Q3 = 145',legend= False,figsize=(20, 10))

plt.show()
pd.DataFrame(list(higest_earned_movies.vote_average), index=list(higest_earned_movies.original_title)).plot(kind='bar',

title='Vote Average of Highest earned movies; Q3=6.6 ',legend= False,figsize=(20, 10))

plt.show()
def movies_highest_genre_count(data):

    """

        This function takes a Data and returns a dictionary having

        genre as key and count of genre as value.

    """

    movies_highest_by_genre = {} # declare of dictionary

    for genres in data:

        for genre in genres:

            if genre in movies_highest_by_genre:  #check genre is exist

                movies_highest_by_genre[genre] += 1 # increment genre if it is exist

            else:

                movies_highest_by_genre[genre] = 1  # new genre entry 

           

                

    return movies_highest_by_genre
highest_revenue_genres = movies_highest_genre_count(higest_earned_movies.genres)

#highest_revenue_genres
pd.DataFrame(highest_revenue_genres, index=['Genres']).plot(kind='bar',

title='Frequency of Genres of Highest earned movies',figsize=(20, 10))

plt.show()
# create dataframe contains only revenue adj >0

df_revenue = df_tmdb[df_tmdb.revenue_adj > 0].sort_values(by='release_year',ascending=True)

index_list = list(range(0, df_revenue.shape[0]))

df_revenue.index = index_list # rearrange the index 

#df_revenue.head(50)
ma_df_revenue=[]

ma_df_revenue_year=[]

t_revenue = 0



for i in range(df_revenue.shape[0]-31):

    for j in range(31):

        t_revenue += df_revenue['revenue_adj'][i+j] 

    

    ma_df_revenue.append(round(t_revenue/30,2))

    ma_df_revenue_year.append(df_revenue['release_year'][i+30])

# create dataframe contains only budget adj >0

df_budget = df_tmdb[df_tmdb.budget_adj > 0].sort_values(by='release_year',ascending=True)

index_list = list(range(0, df_budget.shape[0])) # same size to revenue

df_budget.index = index_list # rearrange the index 
ma_df_budget =[]

t_budget  = 0



for i in range(df_revenue.shape[0]-31):

    for j in range(30):

        t_budget += df_budget['revenue_adj'][i+j] 

    

    ma_df_budget.append(round(t_budget/30,2))

plt.figure(figsize=(20,10))

plt.plot(ma_df_revenue_year,ma_df_budget,color='orange')

plt.plot(ma_df_revenue_year,ma_df_revenue,color='green')

plt.xlabel('Years')

plt.ylabel('Amount in $')

plt.title('Budget vs Revenue 1960 to 2015 ')

plt.legend(['Budget','Revenue'])

plt.show()
def calculate_actors_gross_revenue(movies):   

    """

        This function takes data, forming dictionary having actors as key and sum of earned revenue as value. 

        return this formed dictionary to the caller.

    """

    actors_gross_revenue ={}

    for id,row in movies.iterrows():

        #print(row)

        for key in row.cast:

            if key in actors_gross_revenue:

                #print(actors_gross_revenue[key])

                actors_gross_revenue[key] +=  row.revenue_adj 

            else:

                actors_gross_revenue[key] = row.revenue_adj 

    

    return actors_gross_revenue;

top_actors_gross_revenue = sorted(calculate_actors_gross_revenue(df_revenue).items(), key=lambda item: item[1], reverse=True)[:10]
def list_to_dict(data, label):

    """

        This function uses data as value and label as key for froming a dictionary. At the end return this dictionary.

    """

    top_actors = {label: []}

    index = []

    

    for item in data:

        top_actors[label].append(item[1])

        index.append(item[0])

        

    return top_actors, index



top10_actors_revenue, top10_actors_name = list_to_dict(top_actors_gross_revenue, 'actors')



pd.DataFrame(top10_actors_revenue, index=top10_actors_name).plot(kind='bar',title='Top 10 actors gross revenue all time',figsize=(20, 15))

plt.show()
pd.DataFrame(top10_actors_revenue, index=top10_actors_name).plot(kind='pie',y='actors',title='Top 10 actors gross revenue all time',figsize=(20, 15))

plt.show()