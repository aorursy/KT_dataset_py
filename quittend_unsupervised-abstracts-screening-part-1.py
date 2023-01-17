!pip install fastcluster
import re

import gc

import fastcluster

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as patches



from itertools import chain

from scipy.stats import entropy

from scipy.spatial.distance import squareform

from scipy.cluster.hierarchy import fcluster

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import pairwise_distances

from sklearn.neighbors import KNeighborsClassifier

from spacy import displacy



RANDOM_STATE = 3472



np.random.seed(RANDOM_STATE)



%matplotlib inline
df = pd.read_csv('/kaggle/input/cleaning-cord-19-metadata/cord_metadata_cleaned.csv')



# A simplification algorithm returns NaN for somes texts => no useful information

df = df.dropna(subset=['text_simplified']).reset_index(drop=True)
tokens = df['text_simplified'].str.split(' ').tolist()

tokens = pd.Series(chain(*tokens))

tokens_count = tokens.value_counts()

tokens_count
ax = tokens_count.plot(figsize=(15, 5))

ax.set_xlabel("Token")

ax.set_ylabel("Count")

ax.grid(True)
ax = tokens_count[10000:].plot(figsize=(15, 5))

ax.set_xlabel("Token")

ax.set_ylabel("Count"),

ax.grid(True)
tfidf_vectorizer = TfidfVectorizer(

    input='content',

    lowercase=False,

    preprocessor=lambda text: text,  

    tokenizer=lambda text: text.split(' '),

    token_pattern=None,

    analyzer='word',

    stop_words=None,

    ngram_range=(1, 1),

    max_features=10000,

    binary=False,

    norm='l2',

    use_idf=True,

    smooth_idf=True,

    sublinear_tf=False,

)



features = tfidf_vectorizer.fit_transform(df['text_simplified'])

features.shape
features = features.astype('float32').toarray()
sample_size = 0.1

sample_mask = np.random.choice(

    a=[True, False], 

    size=len(features), 

    p=[sample_size, 1 - sample_size]

)



features_sample = features[sample_mask]

features_sample.shape
%%time

distance_matrix = pairwise_distances(features_sample, metric='cosine')
%%time 

distances = squareform(distance_matrix, force='tovector')

Z = fastcluster.linkage(distances, method='complete', preserve_input=True)
sns.clustermap(

    data=distance_matrix,

    col_linkage=Z, 

    row_linkage=Z,

    cmap=plt.get_cmap('RdBu'),

)
dissimilarities = pd.Series(distance_matrix.flatten())
ax = dissimilarities.hist(bins=100, figsize=(15, 5))

ax.set_xlabel("Cosine dissimilarity")

ax.set_ylabel("Count")

ax.grid(True)
ax = dissimilarities[dissimilarities >= 0.8].hist(bins=100, figsize=(15, 5))

ax.set_xlabel("Cosine dissimilarity")

ax.set_ylabel("Count")

ax.grid(True)
ax = dissimilarities[dissimilarities >= 0.95].hist(bins=100, figsize=(15, 5))

ax.set_xlabel("Cosine dissimilarity")

ax.set_ylabel("Count")

ax.grid(True)
# Cluster features

clusters = fcluster(Z, t=0.999, criterion='distance')



# Plot clustermap

clustermap = sns.clustermap(

    data=distance_matrix,

    col_linkage=Z, 

    row_linkage=Z,

    cmap=plt.get_cmap('RdBu'),

)



# Draw clusters on the clustermap plot

cluster_mapping = dict(zip(range(len(features_sample)), clusters))

clustermap_clusters = pd.Series(

    [cluster_mapping[id_] for id_ in list(clustermap.data2d.columns)]

)



for cluster in set(clusters):

    cluster_range = list(clustermap_clusters[clustermap_clusters == cluster].index)

    clustermap.ax_heatmap.add_patch(

        patches.Rectangle(

            xy=(np.min(cluster_range), np.min(cluster_range)), 

            width=len(cluster_range), 

            height=len(cluster_range),

            fill=False,

            edgecolor='lightgreen',

            lw=2

        )

    )

    

print(f'There are {clustermap_clusters.nunique()} clusters.')
del clustermap

del distance_matrix

del distances

del Z



gc.collect()
model = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)

model.fit(features_sample, clusters)
df['cluster'] = model.predict(features)
cluster_count = df['cluster'].value_counts().sort_values()



ax = cluster_count.plot(kind='bar', figsize=(15, 5))

ax.set_xticks([])

ax.set_xlabel("Cluster id")

ax.set_ylabel("Count")

ax.grid(True)
noise_clusters = set(cluster_count[cluster_count <= 5].index)

noise_mask = df['cluster'].isin(noise_clusters)



df.loc[noise_mask, 'cluster'] = -1
cluster_count = df['cluster'].value_counts().sort_values()



ax = cluster_count.plot(kind='bar', figsize=(15, 5))

ax.set_xticks([])

ax.set_xlabel("Cluster id")

ax.set_ylabel("Count")

ax.grid(True)
columns = np.array(tfidf_vectorizer.get_feature_names())

top_k = 3



def describe(df: pd.DataFrame) -> pd.DataFrame:

    order = features[df.index].mean(axis=0).argsort()[::-1][:top_k]

    top_words = columns[order]

    

    cluster_id = df['cluster'].iloc[0]

    for i, word in enumerate(top_words):

        # For noisy clusters don't use keywords!

        df[f'word_{i + 1}'] = word if cluster_id != -1 else ''

        

    return df



df = df.groupby('cluster').apply(describe)
df.filter(regex='text_simplified|word_\d+', axis=1)
cluster_id = 10

df_cluster = df.loc[df['cluster'] == cluster_id, :]



keywords = (df

    .loc[df['cluster'] == 10, ['word_1', 'word_2', 'word_3']]

    .drop_duplicates()

    .values

    .tolist()[0]

) 



keywords
elements = []

for _, text in df_cluster['text_simplified'].items():

    ents = []

    text = '• ' + text

    for keyword in keywords:

        matches = list(re.finditer(keyword, text))

        for match in matches:

            start, end = match.span()

            ents.append({

                "start": start,

                "end": end, 

                "label": 'KEYWORD'

            })

            

    elements.append({

        'text': text,

        "ents": ents,

    })
displacy.render(elements, style="ent", jupyter=True, manual=True)
(df

    .drop(columns=['title_lang', 'abstract_lang', 'distance'])

    .to_csv('/kaggle/working/cord_metadata_keywords.csv', index=False)

)