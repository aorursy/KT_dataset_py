import pandas as pd    # Importing Pandas library
import numpy as np     # Importing Numpy library

#Importing data
df1 = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv") 
df2 = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")

df1.head(5)
df2.head(5)
df1.columns = ['id','tittle','cast','crew']
df = df2.merge(df1,on='id')
df.head(5)
print (df.columns)
c = df['vote_average'].mean()
print (c)
m = df['vote_count'].quantile(0.95)
qualified_movies = df.copy().loc[df['vote_count']>=m]
qualified_movies.shape
def weighted_rating(x):
    v = x['vote_count']
    r = x['vote_average']
    return (v/(v+m) *r) + (m/(v+m) *c)
qualified_movies['score'] = qualified_movies.apply(weighted_rating,axis=1)
#Sort movies based on score calculated above
qualified_movies = qualified_movies.sort_values('score', ascending=False)

#Print the top 10 movies
qualified_movies[['title','vote_count','vote_average','score']].head(10)
df['overview'].head(5)
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['overview'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape

# Import linear_kernel from sklearn
from sklearn.metrics.pairwise import linear_kernel

#Compute cosine similarity using tfidf_matrix
cosine_sim = linear_kernel(tfidf_matrix , tfidf_matrix)


#Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
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
    return df2['title'].iloc[movie_indices]
get_recommendations('The Dark Knight Rises')
# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast','crew','keywords','genres']

for i in features:
    df[i] = df[i].apply(literal_eval)



# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(n):
    for i in n:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
#Function for getting top 3 elements from the list
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
df['director'] = df['crew'].apply(get_director)
features = ['cast','keywords','genres']
for i in features:
    df[i] = df[i].apply(get_list)
#Print the new features
df[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
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

#Applying clean data function

features = ['cast','keywords','genres','director']

for i in features:
    df[i] = df[i].apply(clean_data)
def create_soup(x):
    return ' '.join(x['keywords'])+' '+' '.join(x['cast'])+' '+' '.join(x['director'])+' '+' '.join(x['genres'])
df['soup'] = df.apply(create_soup,axis = 1)

# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words = 'english')
cv_matrix = count.fit_transform(df['soup'])
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(cv_matrix,cv_matrix)

indices = pd.Series(df.index,index=df['title'])

#Now we can use get_recommendations with cosine_sim2 for metadata based recommender
get_recommendations('The Avengers',cosine_sim2)
from surprise import SVD,Dataset,Reader,accuracy
from surprise.model_selection import cross_validate,KFold
reader = Reader()
ratings = pd.read_csv("../input/the-movies-dataset/ratings_small.csv")
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()
svd.fit(trainset)
ratings[ratings['userId'] == 1]
svd.predict(1, 302, 3)