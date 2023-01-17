import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from wordcloud import WordCloud
import os
movie = pd.read_csv('../input/movie-recommendation-engine/movies.csv')
ratings = pd.read_csv('../input/movie-recommendation-engine/ratings.csv')
movie.head()
ratings.head()
movie.info()
print("___________.......................______________")
ratings.info()
# Dimension of datasets:
print("Dimension of Movie Dataset is")
movie.shape
print("Dimension of Ratings Dataset is")
ratings.shape
movie.describe()
ratings.describe()
genres=[]
for genre in movie.genres:
    
    x=genre.split('|')
    for i in x:
         if i not in genres:
            genres.append(str(i))
genres=str(genres)    
movie_title=[]
for title in movie.title:
    movie_title.append(title[0:-7])
movie_title=str(movie_title)    
# Format of both the cloud 
wordcloud_genre=WordCloud(width=1500,height=800,background_color='black',min_font_size=2,min_word_length=3).generate(genres)
wordcloud_title=WordCloud(width=1500,height=800,background_color='cyan',min_font_size=2,min_word_length=3).generate(movie_title)
plt.figure(figsize=(30,15))
plt.axis('off')
plt.title('WORDCLOUD for Movies Genre',fontsize=30)
plt.imshow(wordcloud_genre)
plt.figure(figsize=(30,15))
plt.axis('off')
plt.title('WORDCLOUD for Movies title',fontsize=30)
plt.imshow(wordcloud_title)
df = pd.merge(ratings,movie, how='left', on='movieId')
df.head()
df1=df.groupby(['title'])[['rating']].sum()
df1.head(10)
high_rated=df1.nlargest(20,'rating') # by default first 20 rated movies
high_rated.head()
plt.figure(figsize=(30,10))
plt.bar(high_rated.index,high_rated['rating'])
plt.ylabel('ratings', fontsize=20)
plt.xticks(fontsize=20,rotation=90)
plt.xlabel('Movie Title', fontsize=20)
plt.yticks(fontsize=15)
# after viewing the sum let's check the count of the movies rating how many time movies are rated:
df2=df.groupby('title')[['rating']].count()
rating_count_20=df2.nlargest(20,'rating')
rating_count_20.head()
plt.figure(figsize=(30,10))
plt.title('Top 20 movies with highest number of ratings',fontsize=30)
plt.xticks(fontsize=25,rotation=90)
plt.yticks(fontsize=25)
plt.xlabel('movies title',fontsize=30)
plt.ylabel('Ratings Count',fontsize=30)

plt.bar(rating_count_20.index,rating_count_20.rating,color='red')
#term frequency inverse document frequency:
#Define a TF-IDF Vectorizer Object
CV=TfidfVectorizer()

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix=CV.fit_transform(movie['genres'])
#Output the shape of tfidf_matrix
tfidf_matrix.shape
#Creating a pivot table array for our customize view:
movie_user = df.pivot_table(index='userId',columns='title',values='rating')
movie_user.head()
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#Construct a reverse map of indices and movie titles
indices=pd.Series(movie.index,index=movie['title']).drop_duplicates()
indices
# Function that takes in movie title as input and outputs most similar movies
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
    return movie['title'].iloc[movie_indices]

# without explaination:

#titles=movies['title']
#def recommendations(title):
 #   idx = indices[title]
  #  sim_scores = list(enumerate(cosine_sim[idx]))
   # sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   # sim_scores = sim_scores[1:21]
    #movie_indices = [i[0] for i in sim_scores]
    #return titles.iloc[movie_indices]
# # Top 10 Similar movies to Toy Story (1995):
get_recommendations('Toy Story (1995)')
# Top 10 Similar movies to Pulp Fiction (1994):
get_recommendations('Pulp Fiction (1994)')
# Top 10 Similar movies to Jumanji 1995:
get_recommendations('Jumanji (1995)')
get_recommendations('Casino (1995)')