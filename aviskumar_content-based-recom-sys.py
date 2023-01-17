import pandas as pd



df = pd.read_csv('../input/content-based-recomm-sys/movies_metadata.csv')
import os

os.listdir('../input')
df.head()
df['tagline'].fillna('')

df['description'] = df['overview'] + df['tagline']

# df['description'] = df['description'].fillna('')
df.shape
df.dropna(subset=['description'], inplace=True)

df['title'].drop_duplicates(inplace=True)
df.shape
df.reset_index()
from sklearn.feature_extraction.text import TfidfVectorizer



tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

tfidf_matrix = tf.fit_transform(df['description'])

print(tfidf_matrix)
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel



cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_similarities.shape
cosine_similarities[0].shape
# df = df.reset_index()

titles = df['title']

indices = pd.Series(df.index, index=df['title'])
def recommend(title):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_similarities[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    movie_indices = [i[0] for i in sim_scores]

    return titles.iloc[movie_indices]
recommend('The Godfather').head(10)
recommend('The Dark Knight Rises').head(10)