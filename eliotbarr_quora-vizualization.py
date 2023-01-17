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
data = pd.read_csv('../input/train.csv').sample(5000, random_state=123)
print('data shape:', data.shape)
test = pd.DataFrame()

test['text'] = pd.concat([data['question1'],data['question2']], axis=0)

test = test.reset_index()
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
from sklearn.feature_extraction.text import TfidfVectorizer



# min_df is minimum number of documents that contain a term t

# max_features is maximum number of unique tokens (across documents) that we'd consider

# TfidfVectorizer preprocesses the descriptions using the tokenizer we defined above



vectorizer = TfidfVectorizer(min_df=10, max_features=10000, tokenizer=tokenizer, ngram_range=(1, 2))

vz = vectorizer.fit_transform(list(test['text']))
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



from sklearn.cluster import MiniBatchKMeans



num_clusters = 40

kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 

                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)

kmeans = kmeans_model.fit(vz)

kmeans_clusters = kmeans.predict(vz)

kmeans_distances = kmeans.transform(vz)
from sklearn.manifold import TSNE



tsne_model = TSNE(n_components=2, verbose=1, random_state=0)

tsne_kmeans = tsne_model.fit_transform(kmeans_distances)
import bokeh.plotting as bp

from bokeh.models import HoverTool, BoxSelectTool

from bokeh.plotting import figure, show, output_notebook



output_notebook()

plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="tf-idf clustering of the news",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)
import numpy as np



colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",

"#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",

"#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",

"#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79",'#ADFF2F','#FF69B4','#E6E6FA','#00FF00',

                     '#800000','#F0F8FF','#A52A2A','#7FFF00','#00FFFF','#00008B'])



plot_kmeans = bp.figure(plot_width=700, plot_height=600, title="KMeans clustering of the questions",

    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",

    x_axis_type=None, y_axis_type=None, min_border=1)
kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])

kmeans_df['cluster'] = kmeans_clusters

kmeans_df['question'] = test['text']
plot_kmeans.scatter(x='x', y='y', 

                    color=colormap[kmeans_clusters], 

                    source=kmeans_df)

hover = plot_kmeans.select(dict(type=HoverTool))

hover.tooltips={"question": "@question", "cluster":"@cluster"}

show(plot_kmeans)