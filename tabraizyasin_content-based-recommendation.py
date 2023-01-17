# Importing important libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# Loading Data 

movies = pd.read_csv('../input/netflix-shows/netflix_titles.csv')

movies.head()
# A concise summary of a data

movies.info()
# To Avoid White Space Issues

movies['listed_in'] = movies.listed_in.str.replace(', ', '|')

movies['listed_in'] = movies['listed_in'].apply(lambda x: x.strip())

movies.head(5)
# Spliting the listed_in into list listed_in to simplify future use

movies['listed_in'] = movies.listed_in.str.split('|')

movies.head(5)
# Converting the categorical data



moviesDetails = movies.copy()



for index, row in movies.iterrows():

    for genre in row['listed_in']:

        moviesDetails.at[index, genre] = 1

    

moviesDetails = moviesDetails.fillna(0)

moviesDetails.head(5)
# Defining user preferences

userInput = [{'title':'Black Panther', 'ratings':4.6},

            {'title':'Thor: Ragnarok', 'ratings':3.5},

            {'title':'Avengers: Infinity War', 'ratings':5.0},

            {'title':"Marvel's The Defenders", 'ratings':2.0},

            {'title':"Men in Black", 'ratings':4.4}

            ]

inputMovies = pd.DataFrame(userInput)



#Visualizing user preferences

plt.bar(inputMovies['title'], inputMovies["ratings"])



plt.xticks(rotation=45)

plt.xlabel("Title")

plt.ylabel("Ratings")

plt.title("User Preferences")
# Merging the show_id with user preferences

inputId = movies[movies['title'].isin(inputMovies['title'].tolist())]



inputMovies = pd.merge(inputId, inputMovies)

inputMovies = inputMovies.drop('type', 1).drop('director', 1).drop('cast', 1).drop('country', 1).drop('date_added', 1).drop('release_year', 1).drop('rating', 1).drop('duration', 1).drop('listed_in', 1).drop('description', 1)



inputMovies
# Gathering the movies from the input

userMovies = moviesDetails[moviesDetails['show_id'].isin(inputMovies['show_id'].tolist())]

userMovies
# Resetting index and dropping unnecessary features to avoid issues

userMovies = userMovies.reset_index(drop=True)

userGenre = userMovies.drop('show_id', 1).drop('title', 1).drop('type', 1).drop('director', 1).drop('cast', 1).drop('country', 1).drop('date_added', 1).drop('release_year', 1).drop('rating', 1).drop('duration', 1).drop('listed_in', 1).drop('description', 1)

userGenre
inputMovies['ratings']
# Dot produt to get user profile

userProfile = userGenre.transpose().dot(inputMovies['ratings'])

userProfile.head(10)
# Now let's get the genres of every movie in our original dataframe And drop the unnecessary information

detailedTable = moviesDetails.set_index(moviesDetails['show_id'])

detailedTable = detailedTable.drop('show_id', 1).drop('title', 1).drop('type', 1).drop('director', 1).drop('cast', 1).drop('country', 1).drop('date_added', 1).drop('release_year', 1).drop('rating', 1).drop('duration', 1).drop('listed_in', 1).drop('description', 1)

detailedTable.head()
detailedTable.shape
# Multiply the genres by the user profile and then take the weighted average

recommendedMovies = ((detailedTable*userProfile).sum(axis=1))/(userProfile.sum())

recommendedMovies.head()
# Sort our recommendations in descending order to get the peaks at top

recommendedMovies = recommendedMovies.sort_values(ascending=False)

recommendedMovies.head()
#Final Recomendation Table



final_Table = movies.loc[movies['show_id'].isin(recommendedMovies.head(20).keys())]

final_Table.title.to_frame()