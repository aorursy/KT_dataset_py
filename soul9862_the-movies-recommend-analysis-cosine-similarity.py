import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel
data = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')

data.head(5)
data = data.head(10000)
data['overview'].isnull().sum()
data['overview'] = data['overview'].fillna('')
data['overview'].isnull().sum()
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(data['overview'])

print(tfidf_matrix.shape)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data['title']).drop_duplicates()

print(indices)
idx = indices['Miracle in Milan']

print(idx)
def get_recommendations(title, cosine_sim=cosine_sim):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]



    return data['title'].iloc[movie_indices]
get_recommendations('The Frisco Kid')