import os

import pandas as pd
# load data file; please change the path to the data file if needed

df = pd.read_csv("/kaggle/input/covid19-public-media-dataset/covid19_articles.csv")

print(len(df))

df.head()
# aggregate by domains

domain_stats = df.domain.value_counts(ascending=True)

pd.DataFrame(domain_stats).plot.barh(figsize=(8, 8), legend=False)
# aggregate by topic areas

topic_area_stats = df.topic_area.value_counts(ascending=True)

pd.DataFrame(topic_area_stats).plot.barh(figsize=(8, 3), legend=False)
# aggregate dates

date_stats = df.date.value_counts()

date_stats.sort_index(inplace=True)

date_stats.plot()
import spacy

NLP = spacy.load("en_core_web_sm")
RELEVANT_POS_TAGS = ["PROPN", "VERB", "NOUN", "ADJ"]

CUSTOM_STOPWORDS = ["say", "%", "will", "new", "would", "could", "other", 

                    "tell", "see", "make", "-", "go", "come", "can", "do", 

                    "such", "give", "should", "must", "use"]



def preprocess(text):

    doc = NLP(text)

    relevant_tokens = " ".join([token.lemma_.lower() for token in doc if token.pos_ in RELEVANT_POS_TAGS and token.lemma_.lower() not in CUSTOM_STOPWORDS])

    return relevant_tokens
from tqdm import tqdm

tqdm.pandas()

processed_content = df["content"].progress_apply(preprocess)

df["processed_content"] = processed_content
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn import metrics

from scipy.spatial.distance import cdist

from sklearn.manifold import TSNE

import numpy as np
def vectorize(text, maxx_features):    

    vectorizer = TfidfVectorizer(max_features=maxx_features)

    X = vectorizer.fit_transform(text)

    return X
texts = df.processed_content.tolist()

texts[0]
X = vectorize(texts, 2 ** 10)

X.shape
pca = PCA(n_components=0.95, random_state=42)

X_reduced= pca.fit_transform(X.toarray())

X_reduced.shape
distortions = []

K = range(8, 20)

for k in K:

    k_means = KMeans(n_clusters=k, random_state=42).fit(X_reduced)

    k_means.fit(X_reduced)

    distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
import matplotlib.pyplot as plt

X_line = [K[0], K[-1]]

Y_line = [distortions[0], distortions[-1]]



# Plot the elbow

plt.plot(K, distortions, 'b-')

plt.plot(X_line, Y_line, 'r')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('Finding optimal k using the elbow method')

plt.show()
k = 11

kmeans = KMeans(n_clusters=k, random_state=42)

y_pred = kmeans.fit_predict(X_reduced)

df['y'] = y_pred
tsne = TSNE(verbose=1, perplexity=100, random_state=42)

X_embedded = tsne.fit_transform(X.toarray())
%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sns



# sns settings

sns.set(rc={'figure.figsize':(10,10)})



# colors

palette = sns.color_palette("bright", 1)



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)

plt.title('t-SNE without clusters')

plt.savefig("tsne_covid19media_unlabelled.png")

plt.show()
from matplotlib import pyplot as plt

import seaborn as sns



# sns settings

sns.set(rc={'figure.figsize':(10,10)})



# colors

palette = sns.hls_palette(k, l=.4, s=.9)



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)

plt.title('t-SNE with {} clusters'.format(k))

plt.savefig("tsne_covid19media_labelled.png")

plt.show()