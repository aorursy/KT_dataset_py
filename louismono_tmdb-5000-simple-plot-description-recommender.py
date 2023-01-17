#current directory 

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# Load Movies Metadata



movies = pd.read_csv('../input/tmdb_5000_movies.csv', low_memory=False)

credit = pd.read_csv('../input/tmdb_5000_credits.csv', low_memory=False)

# Print the first three rows : movies 

movies.head(3)
# Print the first three rows : credits

credit.head(3)
# Simple Recommender : IMDb Top 250    



# As a first step, let's calculate the value of C, the mean rating across all movies:



# Calculate C



C =  movies['vote_average'].mean()

print(C)





#The average rating of a movie on IMDB is around 6.09, on a scale of 10.

#Next, let's calculate the number of votes, m, received by a movie in the 90th percentile. 

#The pandas library makes this task extremely trivial using the .quantile() method of a pandas Series:





# Calculate the minimum number of votes required to be in the chart, m

m = movies['vote_count'].quantile(0.90)

print(m)



#the minimum number of votes required to be in the in the chart is equal to 1838.
# Filter out all qualified movies into a new DataFrame  q_movies

q_movies = movies.copy().loc[movies['vote_count'] >= m]

q_movies.shape





#There are 481 movies qualified, i.e that have a vote count > 1838  ( 90% percentile)
# Function that computes the weighted rating of each movie

def weighted_rating(x, m=m, C=C):

    v = x['vote_count']

    R = x['vote_average']

    # Calculation based on the IMDB formula

    return (v/(v+m) * R) + (m/(m+v) * C)
# Define a new feature 'score' and calculate its value with `weighted_rating()`

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)



#Sort movies based on score calculated above

q_movies = q_movies.sort_values('score', ascending=False)
#Print the top 10 movies

q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)



#We have to say that this recommender reflect more the popularity audience, 

#since it does not go deep with a better understanding of our attributes such as  overview, 

#keywords, genres , cast or crew. 



#In this section, we  build a system that recommends movies that are similar 

#to a particular movie. More specifically, we will compute pairwise similarity 

#scores for all movies based on their plot descriptions ( overview) and 

#recommend movies based on that similarity score.





#Print plot overviews of the first 5 movies.

movies['overview'].head(5)
'''

In its current form, it is not possible to compute the similarity between any two overviews. To do this, you need to compute the word vectors of each overview or document, as it will be called from now on.



we compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each document. This will give you a matrix where each column represents a word in the overview vocabulary (all the words that appear in at least one document) and each column represents a movie, as before.



In its essence, the TF-IDF score is the frequency of a word occurring in a document, down-weighted by the number of documents in which it occurs. 

This is done to reduce the importance of words that occur frequently in plot overviews and therefore, their significance in computing the final similarity score.



scikit-learn gives us a built-in TfIdfVectorizer class that produces the TF-IDF matrix .

'''



#Import TfIdfVectorizer from scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer



#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')





#Replace NaN with an empty string

movies['overview'] = movies['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(movies['overview'])





#Output the shape of tfidf_matrix

tfidf_matrix.shape





#tfidf_matrix.todense()



# we see that over 20978 different words were used to describe the 4803 movies in our dataset.

#Since we have used the TF-IDF vectorizer, 

#calculating the dot product will directly give you the cosine similarity score. 

#we use sklearn's linear_kernel() instead of cosine_similarities() since it is faster.





# Import linear_kernel

from sklearn.metrics.pairwise import linear_kernel



# Compute the cosine similarity matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim
# recommender model creation  



'''

we are going to define a function that takes in a movie title as an input and outputs 

a list of the 10 most similar movies. Firstly, we need a reverse mapping of movie titles 

and DataFrame indices. In other words, we need a mechanism to identify the index of a movie in the

movie DataFrame, given its title.

'''



#Construct a reverse map of indices and movie titles

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

indices
# Function that takes in movie title as input and outputs most similar movies

# The steps are explained in Team38.pdf3



def get_recommendations(title, cosine_sim=cosine_sim):

    

    # Get the index of the movie that matches the title

    idx = indices[title]



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]





    # Return the top 10 most similar movies

    return movies['title'].iloc[movie_indices]
get_recommendations('Pulp Fiction') 
get_recommendations('Titanic') 
get_recommendations('The Dark Knight Rises')
#we observe that the system has done a decent job of finding movies

#with similar plot descriptions, however the quality of the recommendations

#can be ameliorate.



#For example ,"The Dark Knight Rises" returns all Batman movies while it's more

#likely that the people who liked that movie are more inclined to enjoy other Christopher Nolan

#movies. This is something that cannot be captured by the present recommender. Thats why we are going

#to investigate  other metadata such as keywords, genres and credits  to build an improved recommender system.


#It goes without saying that the quality of your recommender would be increased with the usage of better metadata. 

#That is exactly what you are going to do in this section. 

#You are going to build a recommender based on the following metadata: 

#the 3 top actors, the director, related genres and the movie plot keywords. 

credit.info()

# Convert IDs to int. Required for merging



credit.rename(columns={'movie_id': 'id'}, inplace=True)



credit['id'] = credit['id'].astype('int')

movies['id'] = movies['id'].astype('int')
# Merge credits into your main movie dataframe



movies_full = movies.merge(credit, on=['id' , 'title'])



movies_full.info()

movies_full.describe()
#From the new features, cast, crew and keywords, we need to extract the three most 

#important actors, the director and the keywords associated with that movie.

#Right now, our data is present in the form of "stringified" lists. 

#we need to convert them into a form that is usable .



# Parse the stringified features into their corresponding python objects

from ast import literal_eval



features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    movies_full[feature] = movies_full[feature].apply(literal_eval)



#Next, we write functions that will help us to extract the required information 

#from each feature. First, we'll import the NumPy package to get access to its NaN constant.

#Next, we can use it to write the get_director() function:





# Import Numpy 

import numpy as np



# Get the director's name from the crew feature. If director is not listed, return NaN

def get_director(x):

    for i in x:

        if i['job'] == 'Director':

            return i['name']

    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.

    

def get_list(x):

    if isinstance(x, list):

        names = [i['name'] for i in x]

        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.

        if len(names) > 3:

            names = names[:3]

        return names



    #Return empty list in case of missing/malformed data

    return []
# Define new director, cast, genres and keywords features that are in a suitable form.

movies_full['director'] = movies_full['crew'].apply(get_director)



features = ['cast', 'keywords', 'genres']

for feature in features:

    movies_full[feature] = movies_full[feature].apply(get_list)
# Print the new features of the first 3 films



movies_full[['title', 'cast', 'director', 'keywords', 'genres']].head(3)



'''

The next step would be to convert the names and keyword instances into lowercase and strip all the spaces between them. 

This is done so that your vectorizer doesn't count the Johnny of "Johnny Depp" and "Johnny Galecki" as the same.

After this processing step, the aforementioned actors will be represented as "johnnydepp" and "johnnygalecki" and 

will be distinct to your vectorizer.

'''





# Function to convert all strings to lower case and strip names of spaces

def clean_data(x):

    if isinstance(x, list):

        return [str.lower(i.replace(" ", "")) for i in x]

    else:

        #Check if director exists. If not, return empty string

        if isinstance(x, str):

            return str.lower(x.replace(" ", ""))

        else:

            return ''

# Apply clean_data function to your features.

features = ['cast', 'keywords', 'director', 'genres']



for feature in features:

    movies_full[feature] = movies_full[feature].apply(clean_data)

    
#We are now in a position to create your "metadata soup", which is a string that 

#contains all the metadata that you want to feed to your vectorizer (namely actors, director and keywords).    



movies_full.head(10)



def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])



# Create a new soup feature

movies_full['soup'] = movies_full.apply(create_soup, axis=1)    



movies_full[['title','director', 'soup']].head(3)
'''

The next steps are the same as what we did with our plot description based recommender. 

One important difference is that you use the CountVectorizer() instead of TF-IDF. 

This is because you do not want to down-weight the presence of an actor/director 

if he or she has acted or directed in relatively more movies. 

It doesn't make much intuitive sense.

'''





# Import CountVectorizer and create the count matrix

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(movies_full['soup'])





# Compute the Cosine Similarity matrix based on the count_matrix

from sklearn.metrics.pairwise import cosine_similarity



cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

cosine_sim2
# Reset index of your main DataFrame and construct reverse mapping as before

movies_full = movies_full.reset_index() 

indices = pd.Series(movies_full.index, index=movies_full['title'])





movies_full[['title','genres']].head(5)
get_recommendations('Titanic', cosine_sim2)
#We can now reuse your get_recommendations() function by passing in the new cosine_sim2 matrix as your second argument.



get_recommendations('The Dark Knight Rises', cosine_sim2)



#we observe that our recommender has been successful in capturing more information due to more metadata and has given us (arguably) better recommendations.

#for example, the recommender for the Dark knight rises doesn't output all the Batman movies anymore. 

#The recommendations seem to have recognized other Christopher Nolan movies (due to the high weightage given to director) and put them as top recommendations. 





#One thing that we notice about our recommendation system is that it recommends movies regardless of ratings and popularity.

#It is true that Catwoman has a lot of similar characters as compared to The Batman begins but it was a terrible movie that shouldn't be recommended to anyone.

#Many critics consider it to be one of the worst films of all time.(https://en.wikipedia.org/wiki/Catwoman_(film)#cite_note-4)



#Therefore, we will add a mechanism to remove bad movies and return movies which are popular and have had a good critical response.



#I will take the top 25 movies based on similarity scores and calculate the vote of the 60th percentile movie. 

#Then, using this as the value of $m$, we will calculate the weighted rating of each movie using IMDB's formula like we did in the Simple Recommender section.

movies_full.info()
pd.options.mode.chained_assignment = None



def improved_recommendations(title):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim2[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:26]

    movie_indices = [i[0] for i in sim_scores]

    

    movie = movies_full.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'release_date']]

    vote_counts = movie[movie['vote_count'].notnull()]['vote_count'].astype('int')

    vote_averages = movie[movie['vote_average'].notnull()]['vote_average'].astype('int')

    C = vote_averages.mean()

    m = vote_counts.quantile(0.60)

    qualified = movie[(movie['vote_count'] >= m) & (movie['vote_count'].notnull()) & (movie['vote_average'].notnull())]

    qualified['vote_count'] = qualified['vote_count'].astype('int')

    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(weighted_rating, axis=1)

    qualified = qualified.sort_values('wr', ascending=False).head(10)

    return qualified
improved_recommendations('Titanic')
improved_recommendations('The Dark Knight Rises')


#Now we can see that the recommendations seem to have recognized not only other Christopher Nolan movies , not only fill with 

#the 3 top actors, the director, related genres and the movie plot keywords characteristics,  but also the popularity of the movie.



#you can find these ideas implemented in the following github repo : https://github.com/rounakbanik/movies/blob/master/movies_recommender.ipynb