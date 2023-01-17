import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import networkx as nx
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import altair as alt
import plotly
import os
from math import sqrt

%precision %.2f
pd.options.display.float_format = '{:,.2f}'.format

#print(os.listdir("../input"))
credits = pd.read_csv('../input/tmdb_5000_credits.csv')
movies = pd.read_csv('../input/tmdb_5000_movies.csv')
movies.head(1)
credits.head(1)
genre_list = movies.genres.apply(json.loads).apply(lambda x: [e['name'] for e in x if 'name' in e])
pd.DataFrame(genre_list[0])
unique_genres = set([])
for x in genre_list.values:
    for e in x:
        unique_genres.add(e)
#len(list(unique_genres))
pd.DataFrame(list(unique_genres))
def build_gender_row(genre_list, all_genres=unique_genres):
    row_movie_gender = pd.Series(0, index=all_genres)
    row_movie_gender[genre_list]=1
    return row_movie_gender 
genres = pd.DataFrame([build_gender_row(e) for e in genre_list])
#for movie_gender in genre_list
#genres = pd.concat([movies['original_title'], genres], axis = 1)
genres.head(1)
genres.Family.sum()
genres.Animation.sum()
genres.query('Family == 1 & Animation == 1').shape[0]
(genres.query('Family == 1 & Animation == 1').shape[0])\
/(sqrt(genres.Family.sum())*sqrt(genres.Animation.sum()))
genres_sim = pd.DataFrame(cosine_similarity(genres.T))
genres_sim.columns = genres.columns
genres_sim.index = genres.columns
genres_sim
df = genres_sim.where(np.triu(np.ones(genres_sim.shape), 1).astype(np.bool))
df = df.stack().reset_index()
df.columns = ['Row','Column','Value']
df.sort_values('Value', ascending = False).head(10)
df.query('Row == "Adventure" | Column == "Adventure"').sort_values('Value', ascending = False).head(10)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
init_notebook_mode(connected=True) #do not miss this line

names = genres_sim.columns.tolist()
dendro = ff.create_dendrogram(genres_sim, orientation='left', labels=names)
dendro['layout'].update({'width':800, 'height':600, 'margin':go.layout.Margin(
        l=150,
        r=50,
        b=50,
        t=50,
        pad=0
    )})

py.offline.iplot(dendro)
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

Z = hierarchy.linkage(genres.T, 'complete', 'euclidean')
# single, complete, average, weighted, centroid, ward, median
# cosine, euclidean, jaccard, etc.

fig = plt.figure()
fig.set_size_inches(12, 10)
names = genres_sim.columns.tolist()
dn = hierarchy.dendrogram(Z, orientation='right', labels=names)
G = nx.from_pandas_adjacency(genres_sim)
G.name = 'Graph from pandas adjacency matrix'
print(nx.info(G))
from networkx.algorithms import community
#G = nx.barbell_graph(5, 1)
communities_generator = community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
sorted(map(sorted, next_level_communities))
from sklearn.cluster import AffinityPropagation
import numpy as np

clustering = AffinityPropagation().fit(genres_sim)
clustering 
cluster_centers_indices = clustering.cluster_centers_indices_
labels = clustering.labels_

n_clusters_ = len(cluster_centers_indices)
print(n_clusters_)

# Seems to work just fine
aux = pd.DataFrame(clustering.labels_)
aux.index = genres.columns
aux.sort_values(0)
aux3 = pd.DataFrame(clustering.cluster_centers_)
aux3.columns = genres.columns
aux3
from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(genres.T)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(genres.T)
aux2 = pd.DataFrame(kmeans.labels_)
aux2.index = genres.columns
aux2.sort_values(0)
pd.DataFrame(kmeans.cluster_centers_)
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(affinity='euclidean', linkage='complete', n_clusters=5, connectivity=genres_sim)
clustering.fit(genres.T)
clustering

# If connectivity matrix is not provided the results changes slightly. If linkage is not 'complete' the results are much worse. Other than that, ok.
clustering.labels_
aux2 = pd.DataFrame(clustering.labels_)
aux2.index = genres.columns
aux2.sort_values(0)