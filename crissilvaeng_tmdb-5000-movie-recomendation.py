import json



import numpy as np

import pandas as pd

import seaborn as sns



from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN



from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler



from scipy.cluster.hierarchy import dendrogram, linkage
np.random.seed(42)
def extract_genres(data, key='name', separator='|'):

    elements = json.loads(data)

    targets = []

    for element in elements:

        targets.append(element[key])

    return separator.join(targets)
df = pd.read_csv('../input/tmdb_5000_movies.csv')

df['genres'] = df['genres'].apply(extract_genres)



df.head()
genres = df['genres'].str.get_dummies()

genres.head()
standard_scaler = StandardScaler()

scaled_genres = standard_scaler.fit_transform(genres)

genres[genres.columns] = scaled_genres
movies = pd.concat([df['original_title'], genres], axis=1)
movies[movies['original_title'].isin(['Toy Story', 'Inception', 'Avatar'])]
def kmeans(n_clusters, genres):

    model = KMeans(n_clusters=n_clusters)

    model.fit(genres)

    return [n_clusters, model.inertia_]



elbow_data = [kmeans(n_clusters, scaled_genres) for n_clusters in range(1, 41)]

elbow_data = pd.DataFrame(elbow_data, columns=['clusters', 'inertia'])



elbow_data['inertia'].plot(xticks=elbow_data['clusters'])
model = KMeans(n_clusters=12)

model.fit(scaled_genres)
genres_groups = pd.DataFrame(model.cluster_centers_, columns=genres.columns)

genres_groups
genres_groups.transpose().plot.bar(

    subplots=True,

    figsize=(25,50),

    sharex=False,

    rot=0)
df[model.labels_ == 0].head()
tsne = TSNE()

reduced_genres = tsne.fit_transform(scaled_genres)
sns.set(rc={ 'figure.figsize': (13,13) })

sns.scatterplot(

    x=reduced_genres[:, 0],

    y=reduced_genres[:, 1],

    hue=model.labels_,

    palette=sns.color_palette('Set1', 12))
model = AgglomerativeClustering(n_clusters=12)

agglomerate_genres = model.fit_predict(scaled_genres)
sns.scatterplot(

    x=reduced_genres[:, 0],

    y=reduced_genres[:, 1],

    hue=agglomerate_genres)
distance_matrix = linkage(genres_groups)

distance_matrix
dendrogram(distance_matrix)
model = DBSCAN()

density_genres_clusters = model.fit_predict(scaled_genres)
sns.scatterplot(

    x=reduced_genres[:, 0],

    y=reduced_genres[:, 1],

    hue=density_genres_clusters)