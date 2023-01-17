import pandas as pd

import numpy as np



# genome_scores = pd.read_csv("../input/movielens-20m-dataset/genome_scores.csv")

# genome_tags = pd.read_csv("../input/movielens-20m-dataset/genome_tags.csv")

# link = pd.read_csv("../input/movielens-20m-dataset/link.csv")

movie = pd.read_csv("../input/movielens-20m-dataset/movie.csv")

rating = pd.read_csv("../input/movielens-20m-dataset/rating.csv")

# tag = pd.read_csv("../input/movielens-20m-dataset/tag.csv")
rating.head()
movie = movie.set_index('movieId')

movie['total_rating'] = rating.movieId.value_counts()

movie['mean_rating'] = rating.groupby('movieId').mean()['rating']

movie.head()
import plotly.express as px

data = px.data.gapminder()



fig = px.bar(

    movie[:10].sort_values('total_rating', ascending=False),

    x='title',

    y='total_rating',

    color='mean_rating',

    height=600,

    title='Top 10 - Popular movies',

    labels={'total_rating':'Total number of ratings','mean_rating': 'Mean rating', 'title': 'Titles'}

)

fig.show()
# to calc the distance of two vectors

def distance(x, y):

    return np.linalg.norm(x - y)



# to get all ratings from an user

def ratings_by_user(user_id):

    user_r = rating[ rating.userId == user_id ]

    return user_r.drop(['userId', 'timestamp'], axis=1)



# to find the movies which both watched.

def merge(df1,df2):

    return df1.merge(df2, how='inner', on="movieId")



# to calc the distance of two users

def distance_between_users(user_id1, user_id2):

    vector_user1 = ratings_by_user(user_id1)

    vector_user2 = ratings_by_user(user_id2)

    vectors = merge(vector_user1, vector_user2)

    if vectors.empty:

        return (user_id1, user_id2, 100)

    return user_id1, user_id2, distance(vectors['rating_x'], vectors['rating_y'])



# get all users ids...

def get_users_ids():

    return list(rating.userId.unique())



# passing the user id for reference, calc the distance from all, note I'm limiting this process for only 10 users...

def center_from_all(user_id):

    if not user_id in rating.userId:

        return False   

    

    ids = get_users_ids()

    ids.remove(user_id)

    return [distance_between_users(user_id, for_id) for for_id in ids[:10]]
# returning  movies information

def ids_to_movie(movieId):

    movie.loc[movies_recommend(1)].sort_values('mean_rating', ascending=False)

    

# In this function, I use all the functions declared before to make the magic happen

def movies_recommend(userId, k=10):

    points = center_from_all(userId)

    points = pd.DataFrame(points, columns=['myId','userId','distance'])

    points.sort_values('distance', axis=0, ascending=True, inplace=True)

    movies_list = list()



    my_ratings = ratings_by_user(userId)

    our_users = list(points.userId)



    for our_user in our_users:

        merged = my_ratings.merge(ratings_by_user(our_user), how='right', on="movieId")

        merged = merged[merged.rating_x.isna()]

        movies_list.append(merged)

        movies_list = pd.concat(movies_list)

        return movie.loc[movies_list.movieId].sort_values('mean_rating', ascending=False)[:k]

        

movies_recommend(1)