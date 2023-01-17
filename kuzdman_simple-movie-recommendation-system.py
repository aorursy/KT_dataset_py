import pandas as pd



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity



from surprise import Reader, Dataset, SVD
movies = pd.read_csv('../input/movielens-dataset/movies.csv')
movies.head()
movies['genres'] = movies['genres'].apply(lambda x: x.replace('|', ' '))
movies.head()
count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')

count_matrix = count.fit_transform(movies['genres'])
count_matrix.shape
cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim.shape
movies = movies.reset_index()

titles = movies['title']

indices = pd.Series(movies.index, index=movies['title'])
def get_recommendations(title):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    movie_indices = [i[0] for i in sim_scores]

    return titles.iloc[movie_indices]
get_recommendations('Toy Story (1995)').head(5)
ratings = pd.read_csv('../input/movielens-dataset/ratings.csv')
ratings.head()
reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
trainset = data.build_full_trainset()

svd.fit(trainset)
svd.predict(1, 101).est