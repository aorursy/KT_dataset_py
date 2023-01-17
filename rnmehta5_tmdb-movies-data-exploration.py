import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction import text

from sklearn.decomposition import LatentDirichletAllocation as LDA



%matplotlib inline 
movies = pd.read_csv("../input/tmdb_5000_movies.csv")
movies['allWords'] = movies.overview + " " + movies.tagline

print(movies.loc[0, 'allWords'])
movies = movies[~(movies.overview.isnull()) & ~(movies.tagline.isnull())][['title', 'allWords']]

movies['movie_id'] = movies.index

movies.shape
additional_stop_words = {'production', 'movie', 'movies', 'film', 'films', 'man', 'story'}

my_stop_words = text.ENGLISH_STOP_WORDS.union(additional_stop_words)

tfidf_vect = text.CountVectorizer(max_df=0.8, stop_words=my_stop_words)

tfidf_movies = tfidf_vect.fit_transform(movies['allWords'])

tfidf_feature_names = tfidf_vect.get_feature_names()
num_topics = 5



lda = LDA(n_components=num_topics, learning_method='online').fit(tfidf_movies)
for topic_idx, topic in enumerate(lda.components_):

    print("Topic", topic_idx, ":")

    print(" ".join([tfidf_feature_names[i] for i in topic.argsort()[:-14:-1]]))
doc_topic = lda.transform(tfidf_movies)

for n in range(10):

    curr_topic = doc_topic[n].argmax()

    print("the movie ", movies.loc[n, 'title'], ' belongs to topic', curr_topic)