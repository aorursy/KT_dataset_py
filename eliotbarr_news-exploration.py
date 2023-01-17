# import packages

import requests

import pandas as pd

from datetime import datetime

from tqdm import tqdm

from matplotlib import pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

# list of stopwords like articles, preposition

from string import punctuation

from collections import Counter

import re

stop = set(stopwords.words('english'))
data = pd.read_csv("../input/uci-news-aggregator.csv",nrows=10000)
data.shape
print('data shape:', data.shape)
data.CATEGORY.value_counts().plot(kind='bar', grid=True, figsize=(16, 9))
# remove rows with empty Titles

data = data[~data['TITLE'].isnull()]
data['len'] = data['TITLE'].map(len)
def tokenizer(text):

    tokens_ = [word_tokenize(sent) for sent in sent_tokenize(text)]



    tokens = []

    for token_by_sent in tokens_:

        tokens += token_by_sent



    tokens = list(filter(lambda t: t.lower() not in stop, tokens))

    tokens = list(filter(lambda t: t not in punctuation, tokens))

    tokens = list(filter(lambda t: t not in [u"'s", u"n't", u"...", u"''", u'``', 

                                        u'\u2014', u'\u2026', u'\u2013'], tokens))

    filtered_tokens = []

    for token in tokens:

        if re.search('[a-zA-Z]', token):

            filtered_tokens.append(token)



    filtered_tokens = list(map(lambda token: token.lower(), filtered_tokens))



    return filtered_tokens

data['tokens'] = data['TITLE'].map(tokenizer)
for title, tokens in zip(data['TITLE'].head(5), data['tokens'].head(5)):

    print('title:', title)

    print('tokens:', tokens)

    print() 
def keywords(category):

    tokens = data[data['CATEGORY'] == category]['tokens']

    alltokens = []

    for token_list in tokens:

        alltokens += token_list

    counter = Counter(alltokens)

    return counter.most_common(10)
for category in set(data['CATEGORY']):

    print('category :', category)

    print('top 10 keywords:', keywords(category))

    print('---')
from sklearn.feature_extraction.text import TfidfVectorizer



# min_df is minimum number of documents that contain a term t

# max_features is maximum number of unique tokens (across documents) that we'd consider

# TfidfVectorizer preprocesses the descriptions using the tokenizer we defined above



vectorizer = TfidfVectorizer(min_df=10, max_features=10000, tokenizer=tokenizer, ngram_range=(1, 2))

vz = vectorizer.fit_transform(list(data['TITLE']))
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')

tfidf.columns = ['tfidf']
tfidf.tfidf.hist(bins=50, figsize=(15,7))
tfidf.sort_values(by=['tfidf'], ascending=True).head(30)
tfidf.sort_values(by=['tfidf'], ascending=False).head(30)
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=50, random_state=0)

svd_tfidf = svd.fit_transform(vz)
from sklearn.manifold import TSNE



tsne_model = TSNE(n_components=2, verbose=1, random_state=0)

tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
import bokeh.plotting as bp

from bokeh.models import HoverTool, BoxSelectTool

from bokeh.plotting import figure, show, output_notebook
output_notebook()

plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="tf-idf clustering of the news",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)
tfidf_df = pd.DataFrame(tsne_tfidf, columns=['x', 'y'])

tfidf_df['title'] = data['TITLE']

tfidf_df['category'] = data['CATEGORY']
plot_tfidf.scatter(x='x', y='y', source=tfidf_df)

hover = plot_tfidf.select(dict(type=HoverTool))

hover.tooltips={"title": "@title", "category":"@category"}

show(plot_tfidf)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



from sklearn.cluster import MiniBatchKMeans



num_clusters = 30

kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 

                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)

kmeans = kmeans_model.fit(vz)

kmeans_clusters = kmeans.predict(vz)

kmeans_distances = kmeans.transform(vz)
for (i, desc),category in zip(enumerate(data.TITLE),data['CATEGORY']):

    if(i < 5):

        print("Cluster " + str(kmeans_clusters[i]) + ": " + desc + 

              "(distance: " + str(kmeans_distances[i][kmeans_clusters[i]]) + ")")

        print('category: ',category)

        print('---')
sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

for i in range(num_clusters):

    print("Cluster %d:" % i)

    aux = ''

    for j in sorted_centroids[i, :10]:

        aux += terms[j] + ' | '

    print(aux)

    print() 
tsne_kmeans = tsne_model.fit_transform(kmeans_distances)
import numpy as np



colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",

"#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",

"#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",

"#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])



plot_kmeans = bp.figure(plot_width=700, plot_height=600, title="KMeans clustering of the news",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)
kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])

kmeans_df['cluster'] = kmeans_clusters

kmeans_df['title'] = data['TITLE']

kmeans_df['category'] = data['CATEGORY']
plot_kmeans.scatter(x='x', y='y', 

                    color=colormap[kmeans_clusters], 

                    source=kmeans_df)

hover = plot_kmeans.select(dict(type=HoverTool))

hover.tooltips={"title": "@title", "category": "@category", "cluster":"@cluster"}

show(plot_kmeans)