%matplotlib inline  

# To make data visualisations display in Jupyter Notebooks 

import numpy as np   # linear algebra

import pandas as pd  # Data processing, Input & Output load

import matplotlib.pyplot as plt # Visuvalization & plotting

import seaborn as sns # Also for Data visuvalization 



from sklearn.metrics.pairwise import cosine_similarity  # Compute cosine similarity between samples in X and Y.

from scipy import sparse  #  sparse matrix package for numeric data.



import warnings   # To avoid warning messages in the code run

warnings.filterwarnings("ignore")

Rating = pd.read_csv('../input/Ratings.csv') 

Movie_D = pd.read_csv('../input/Movie details.csv',encoding='latin-1') ##Movie details 

User_Info = pd.read_csv('../input/user level info.csv',encoding='latin-1') ## if you have a unicode string, you can use encode to convert
Rating.shape
Rating.head()
Rating.columns = ['user_id', 'movie_id', 'rating', 'timestamp'] 
Movie_D.shape
Movie_D.head()
Movie_D.columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date ',

       'IMDb_URL', 'unknown', 'Action ', 'Adventure', 'Animation',

       'Childrens', 'Comedy ', 'Crime ', ' Documentary ', 'Drama',

       ' Fantasy', 'Film-Noir ', 'Horror ', 'Musical', 'Mystery',

       ' Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
Movie_Rating = pd.merge(Rating ,Movie_D,on = 'movie_id')

Movie_Rating.describe()
n_users = Movie_Rating.user_id.unique().shape[0]

n_items = Movie_Rating.movie_id.unique().shape[0]

print(n_items,n_users)
# Calculate mean rating of all movies 

Movie_Stats = pd.DataFrame(Movie_Rating.groupby('movie_title')['rating'].mean())

Movie_Stats.sort_values(by = ['rating'],ascending=False).head() 

# Calculate count rating of all movies 

#Movie_Rating.groupby('movie_title')['rating'].count().sort_values(ascending=False).head() 

Movie_Stats['Count_of_ratings'] = pd.DataFrame(Movie_Rating.groupby('movie_title')['rating'].count())

Movie_Stats.sort_values(by =['Count_of_ratings'], ascending=False).head()
Movie_Stats['Count_of_ratings'].hist(bins=50)
sns.jointplot(x='rating', y='Count_of_ratings', data=Movie_Stats)
User_movie_Rating = Movie_Rating.pivot_table(index='user_id', columns='movie_title', values='rating')

User_movie_Rating.head()
##We can achieve this by computing the correlation between these two movies’ ratings and the ratings of the rest of the movies in the dataset. 

##The first step is to create a dataframe with the ratings of these movies 



# Example pick up one movie related rating  

User_movie_Rating['Air Force One (1997)']
Similarity = User_movie_Rating.corrwith(User_movie_Rating['Air Force One (1997)'])

Similarity.head()
corr_similar = pd.DataFrame(Similarity, columns=['Correlation'])

corr_similar.sort_values(['Correlation'], ascending= False).head(10)
corr_similar_num_of_rating = corr_similar.join(Movie_Stats['Count_of_ratings'])

corr_similar_num_of_rating.sort_values(['Correlation'], ascending= False).head(10)
corr_similar_num_of_rating[corr_similar_num_of_rating ['Count_of_ratings']>50].sort_values('Correlation', ascending=False).head()
def get_recommendations(title):

    # Get the movie ratings of the movie that matches the title

    Movie_rating = User_movie_Rating[title]



    # Get the  similarity corrilated  scores of all movies with that movie

    sim_scores = User_movie_Rating.corrwith(Movie_rating)



    # Sort the movies based on the similarity scores

    corr_title = pd.DataFrame(sim_scores, columns=['Correlation'])

     # Removing na values 

    corr_title.dropna(inplace=True)

    

    corr_title = corr_title.join(Movie_Stats['Count_of_ratings'])

    

   ## corr_title[corr_title ['Count_of_ratings']>50].sort_values('Correlation', ascending=False).head()

    



    # Return the top 10 most similar movies

    return corr_title[corr_title ['Count_of_ratings']>50].sort_values('Correlation', ascending=False).head()

    

 
get_recommendations('Air Force One (1997)')
get_recommendations('Star Wars (1977)')
Movie_cosine = Movie_Rating[['user_id','movie_id','rating']]

Movie_cosine.head()
data = Movie_NUM.rating

col = Movie_NUM.movie_id

row = Movie_NUM.user_id



R = sparse.coo_matrix((data, (row, col))).tocsr()

print ('{0}x{1} user by movie matrix'.format(*R.shape))
find_similarities = cosine_similarity(R.T) # We are transposing the matrix 

print (similarities.shape)

#similarities
def Get_Top5_Similarmovies(model, movie_id, n=5):

    return model[movie_id].argsort()[::-1][:n].tolist()  # Here movie id is index
Movie_D.head()
Movie_D.iloc[4] 
Movie_D.iloc[Get_Top5_Similarmovies(find_similarities, 4)]

#similar_movies(similarities, 4)