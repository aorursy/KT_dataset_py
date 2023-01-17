import numpy as np # linear algebra
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
ratings=pd.read_csv('../input/ratings.csv')
ratings.head(15)
movies=pd.read_csv('../input/movies.csv')
movies.head(15)
movie_ratings = pd.merge(movies, ratings)
movie_ratings.head(15)
ratings_matrix = ratings.pivot_table(index=['movieId'],columns=['userId'],values='rating').reset_index(drop=True)
ratings_matrix.fillna( 0, inplace = True )
ratings_matrix.head(15)
movie_similarity=cosine_similarity(ratings_matrix)
np.fill_diagonal( movie_similarity, 0 ) 
movie_similarity
ratings_matrix = pd.DataFrame( movie_similarity )
ratings_matrix.head(15)
try:
    #user_inp=input('Enter the reference movie title based on which recommendations are to be made: ')
    user_inp="Jumanji (1995)"
    inp=movies[movies['title']==user_inp].index.tolist()
    inp=inp[0]
    
    movies['similarity'] = ratings_matrix.iloc[inp]
    movies.head(5)
    
except:
    print("Sorry, the movie is not in the database!")
    
print("Recommended movies based on your choice of ",user_inp ,": \n", movies.sort_values( ["similarity"], ascending = False )[1:10])