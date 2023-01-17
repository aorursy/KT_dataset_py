from subprocess import check_output

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

print(check_output(["ls", "../input/"]).decode("utf8"))

df1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

df2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

df1.columns = ['id','tittle','cast','crew']

df2= df2.merge(df1,on='id')

import seaborn as sns

fig, ax = plt.subplots(figsize=(9,5))

sns.heatmap(df2.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
def correlation_matrix(df):

    from matplotlib import pyplot as plt

    from matplotlib import cm as cm

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    cmap = cm.get_cmap('jet', 50)

    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)

    ax1.grid(True)

    plt.title('Movies Features Correlation')

   # print(df.head(0))

    labels = df.head()

   # labels =["index", "budget", "genres", "homepage", "id", "keywords", "original_language", "original_title", "overview", "popularity", "production_companies", "production_countries", "release_date", "revenue", "runtime", "spoken_languages", "status", "tagline", "title", "vote_average", "vote_count", "tittle", "cast", "crew", "director", "soup"]

    ax1.set_xticklabels(labels,fontsize=10, rotation=40)

    ax1.set_yticklabels(labels,fontsize=10)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels

    fig.colorbar(cax)

    plt.show()
correlation_matrix(df2 )
def get_sperman_correlation(data,cols,col1,col2):

    plt.rcParams['figure.figsize'] = [16, 6]

    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax=ax.flatten()  

    colors=['#415952', '#f35134', '#243AB5', '#243AB5']

    j=0



    for i in ax:

        if j==0:

            i.set_ylabel(ylabel)

        i.scatter(data[cols[j]], data['popularity'],  alpha=0.5, color=colors[j])

        i.set_xlabel(cols[j])

        i.set_title('Pearson: %s'%data.corr().loc[cols[j]][col1].round(2)+' Spearman: %s'%data.corr(method='spearman').loc[cols[j]][col2].round(2))

        j+=1



    plt.show()
cols = ['vote_average', 'popularity', 'vote_count']

ylabel = 'popularity'

get_sperman_correlation(df2[cols],cols, 'vote_average', 'vote_count')
df2.head()
def get_scatter(data):

    import matplotlib.pyplot as plt

    import pandas

    from pandas.plotting import scatter_matrix

    names = data.head()

    scatter_matrix(data)

    plt.show()
get_scatter(df2[['popularity', 'vote_average', 'vote_count']])
sns.jointplot('vote_average', 'popularity', data=df2, kind="hex")

sns.jointplot('vote_average', 'popularity', data=df2, kind="reg")
def hist_plot(df):

    import matplotlib.pyplot as plt

    df2.hist(bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)

    plt.show()
hist_plot(df2[['budget','id','popularity','vote_average']])
#Replace NaN with an empty string

df2['overview'] = df2['overview'].fillna('')
#Construct a reverse map of indices and movie titles

indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

indices.head()
#parse json and separate columns with genres of movie



def endcode_genre(df_movies_genre):

    import json

    import ast

    countGener = df_movies_genre["genres"]

    print(len(countGener))

    for index in range(len(countGener)):

        item = ast.literal_eval(countGener[index])

        for j in item:

            j = str(j).replace("'", '"')

            json_data = json.loads(j)

            name = "genres_" + str(json_data["id"]) + "_" + str(json_data["name"])

            #print(name)

            if {name}.issubset(df_movies_genre.columns):

                df_movies_genre.at[index,name] = 1

            else:

                df_movies_genre[name] = 0

                df_movies_genre.at[index,name] = 1

    return df_movies_genre



#parse json and separate columns with keywords of movie



def endcode_keywords(df2):

    import json

    import ast

    df_movies_keyword = df2[['id','genres','keywords']]

    count = df_movies_keyword["keywords"]

    

    for index in range(len(count)):

       for item in count[index]:

        name = "kw_" + item

        if {name}.issubset(df_movies_keyword.columns):

            df_movies_keyword.at[index,name] = 1

        else:

            df_movies_keyword[name] = 0

            df_movies_keyword.at[index,name] = 1

    return df_movies_keyword



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



C= df2['vote_average'].mean()

C
m= df2['vote_count'].quantile(0.9)

m
q_movies = df2.copy().loc[df2['vote_count'] >= m]

q_movies.shape
def weighted_rating(x, m=m, C=C):

    v = x['vote_count']

    R = x['vote_average']

    # Calculation based on the IMDB formula

    return (v/(v+m) * R) + (m/(m+v) * C)



# Define a new feature 'score' and calculate its value with `weighted_rating()`

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
#Sort movies based on score calculated above

q_movies = q_movies.sort_values('score', ascending=False)



#Print the top 15 movies

q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)
pop= df2.sort_values('popularity', ascending=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))



plt.barh(pop['title'].head(10),pop['popularity'].head(10), align='center',

        color='skyblue')

plt.gca().invert_yaxis()

plt.xlabel("Popularity")

plt.title("Popular Movies")
#Import TfIdfVectorizer from scikit-learn

from sklearn.feature_extraction.text import TfidfVectorizer



#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')



#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(df2['overview'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape

#print(tfidf_matrix)
# Import linear_kernel

from sklearn.metrics.pairwise import linear_kernel



# Compute the cosine similarity matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim[0]
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
get_recommendations('This Thing of Ours')
# Parse the stringified features into their corresponding python objects

from ast import literal_eval



features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:

    df2[feature] = df2[feature].apply(literal_eval)
df2[feature].head(2)
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

df2['director'] = df2['crew'].apply(get_director)



features = ['cast', 'keywords', 'genres']

for feature in features:

    df2[feature] = df2[feature].apply(get_list)
# Print the new features of the first 3 films

df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
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

    df2[feature] = df2[feature].apply(clean_data)
df2.head(2)
def create_soup(x):

    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup, axis=1)
# Import CountVectorizer and create the count matrix

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(df2['soup'])



# Compute the Cosine Similarity matrix based on the count_matrix

from sklearn.metrics.pairwise import cosine_similarity



cosine_sim2 = cosine_similarity(count_matrix, count_matrix)



# Reset index of our main DataFrame and construct reverse mapping as before

df2 = df2.reset_index()

indices = pd.Series(df2.index, index=df2['title'])
cosine_sim2
get_recommendations('The Dark Knight Rises', cosine_sim2)
get_recommendations('The Godfather', cosine_sim2)
import pandas as pd 

import numpy as np 

from matplotlib import pyplot

import warnings

#

warnings.filterwarnings('ignore')

#

%matplotlib inline
def mean(u):

    # may use specified_rating_indices but use more time

    #print(specified_rating_indices(u))

    specified_ratings = u[specified_rating_indices(u)]

    #u[np.isfinite(u)]

    m = sum(specified_ratings)/np.shape(specified_ratings)[0]

    return m



def pearson(u, v):

    mean_u = mean(u)

    mean_v = mean(v)

    

    specified_rating_indices_u = set(specified_rating_indices(u)[0])

    specified_rating_indices_v = set(specified_rating_indices(v)[0])

    

    mutually_specified_ratings_indices = specified_rating_indices_u.intersection(specified_rating_indices_v)

    mutually_specified_ratings_indices = list(mutually_specified_ratings_indices)

    

    u_mutually = u[mutually_specified_ratings_indices]

    v_mutually = v[mutually_specified_ratings_indices]

    

    centralized_mutually_u = u_mutually - mean_u

    centralized_mutually_v = v_mutually - mean_v



    result = np.sum(np.multiply(centralized_mutually_u, centralized_mutually_v)) 

    result = result / (np.sqrt(np.sum(np.square(centralized_mutually_u))) * np.sqrt(np.sum(np.square(centralized_mutually_v))))



    return result





# indices for vector

def specified_rating_indices(u):

    return list(map(tuple, np.where(np.isfinite(u))))



#get similar movies

def get_movie_similarity_value_for(movie_index, movie_matrix):

   # print(movies_matrix.loc[movies_matrix.id == 862])

   

    movie_item = np.array(movie_matrix.loc[movies_matrix.id == 862])

    movie_item = np.delete(movie_item, 0)

    #print(movie_item)

    #print(np.array(movie_matrix.iloc[0, 1:]))

    #print(pearson(movie_matrix.iloc[0, 1:], movie_item))

    similarity_value = np.array([pearson(np.array(movie_matrix.iloc[i, 1:]), movie_item) for i in range(movie_matrix.shape[0])])

    return similarity_value
def getSimilarMovieKeywords(movie_id):

    import ast

    df_movie_meta=pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')

    df_keyword =pd.read_csv('../input/the-movies-dataset/keywords.csv')

    cols = ["id","genres", "title", "overview", "vote_average", "vote_count"]

    df_movie_meta = df_movie_meta[cols]

    df_movie_meta['id'] = df_movie_meta['id'].str.replace('-','')

    df_movie_meta.dropna(subset=["id"], axis = 0 , inplace= True)

    df_movie_meta["id"] = df_movie_meta["id"].astype(str).astype(int)

    df_movie_meta= df_movie_meta.merge(df_keyword,on='id')

    df_movie_meta.set_index('id')



    # Parse the stringified features into their corresponding python objects

    #from ast import literal_eval

    df_movie_meta['keywords'] = df_movie_meta['keywords'].apply(ast.literal_eval)

    df_movie_meta['keywords'] = df_movie_meta['keywords'].apply(get_list)

    #print(df_movie_meta.shape())

    movie_genres_keyword_score = endcode_keywords(df_movie_meta)

    movie_genres_keyword_score = movie_genres_keyword_score.drop(['keywords'], axis=1)

    movie_genres_keyword_score = endcode_genre(movie_genres_keyword_score)

    movie_genres_keyword_score = movie_genres_keyword_score.drop(['genres'], axis=1)

    movie_genres_keyword_score["id"] = movie_genres_keyword_score["id"].astype(str).astype(int)

  

    movie_item = np.array(movie_genres_keyword_score.loc[movie_genres_keyword_score.id == movie_id])

    movie_item = np.delete(movie_item, 0)

    similarity_value = np.array([pearson(np.array(movie_genres_keyword_score.iloc[i, 1:]), movie_item) for i in range(movie_genres_keyword_score.shape[0])])

   

   # print(similarity_value.count())

   # print(df_movie_meta.count())

    df_movie_meta["score"] = similarity_value

    return df_movie_meta, movie_genres_keyword_score
similarMovieList_Keyword_genre, movie_genres_keyword_score =  getSimilarMovieKeywords(28656)

similarMovieList_Keyword_genre = similarMovieList_Keyword_genre[similarMovieList_Keyword_genre.score.notnull()]

similarMovieList_Keyword_genre = similarMovieList_Keyword_genre.sort_values(by='score', ascending=False).head(10)

similarMovieList_Keyword_genre
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="darkgrid")

              

df = movie_genres_keyword_score[movie_genres_keyword_score.columns[pd.Series(movie_genres_keyword_score.columns).str.startswith('genres')]]

df = df.transpose().reset_index().rename(columns={'index':'Genres'})

movie_genres_keyword_score.head()

df['sum'] = df.sum(axis=1)



ax = df.plot(x="Genres", y="sum", kind="bar")

plt.show()
from surprise import Reader, Dataset, SVD, evaluate

reader = Reader()

ratings=pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

data.split(n_folds=5)
svd = SVD()

evaluate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()

svd.fit(trainset)
ratings[ratings['userId'] == 1]
svd.predict(1, 302, 3)


def getMovieListRecommendation(userId):

    import pandas as pd

    import numpy as np



    ratings=pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

    movies_list=pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')



    ratings_df = pd.DataFrame(ratings, columns = ['userId', 'movieId', 'rating', 'timestamp'], dtype = int)

    movies_df = pd.DataFrame(movies_list, columns = ['id', 'title', 'genres'])

    movies_df['id'] = movies_df['id']

    movies_df['id'] = movies_df['id'].str.replace('-','')

    movies_df.dropna(subset=["id"], axis = 0 , inplace= True)

    movies_df["id"] = movies_df["id"].astype(str).astype(int)

    R_df = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating')

    R_df=R_df.fillna(0) 



    #R_df = R_df.fillna(R_df.mean()) # Replace the na with column mean (Movie mean)



    R = R_df.values

    user_ratings_mean = np.mean(R, axis = 1)

    R_demeaned = R - user_ratings_mean.reshape(-1, 1)



    

    from scipy.sparse.linalg import svds

    U, sigma, Vt = svds(R_demeaned, k = 50)



    sigma = np.diag(sigma)



    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)







    def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):



        # Get and sort the user's predictions

        user_row_number = userID

        sorted_user_predictions = predictions_df.iloc[userID].sort_values(ascending=False)

        print (list(pd.DataFrame(sorted_user_predictions).columns))



        # Get the user's data and merge in the movie information.

        user_data = original_ratings_df[original_ratings_df.userId == (userID)]

        user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'id').

                         sort_values(['rating'], ascending=False))



        print ('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))

        print ('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))



        # Recommend the highest predicted rating movies that the user hasn't seen yet.

        recommendations = (movies_df[~movies_df['id'].isin(user_full['movieId'])].

             merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',

                   left_on = 'id',

                   right_on = 'movieId').

             rename(columns = {user_row_number: 'Predictions'}).

             sort_values('Predictions', ascending = False).

                           iloc[:num_recommendations, :-1]

                          )



        return user_full, recommendations





    already_rated, predictions = recommend_movies(preds_df, userId, movies_df, ratings_df, 10)

    return predictions

predictions = getMovieListRecommendation(2)

predictions
from surprise import Dataset, evaluate

from surprise import KNNBasic, Reader, Dataset, SVD, accuracy, Dataset

import pandas as pd

from surprise.model_selection import GridSearchCV



from surprise.model_selection import train_test_split



reader = Reader()

ratings=pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

ratings.head()



data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)



# sample random trainset and testset

# test set is made of 25% of the ratings.

trainset, testset = train_test_split(data, test_size=.25)





#Grid Search CV with SVD



param_grid = {'n_epochs': [15, 30], 'lr_all': [0.002, 0.05],

              'reg_all': [0.1, 0.8]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)



gs.fit(data)



# best RMSE score

print(gs.best_score['rmse'])



# combination of parameters that gave the best RMSE score

print(gs.best_params['rmse'])

#Grid Search CV with KNNBasic



param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],

              'reg_all': [0.4, 0.6]}

gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=3)



gs.fit(data)



# best RMSE score

print(gs.best_score['rmse'])



# combination of parameters that gave the best RMSE score

print(gs.best_params['rmse'])