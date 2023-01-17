# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

from matplotlib import *

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
about_data = pd.read_csv('../input/AboutIsis.csv',encoding = "ISO-8859-1")

fanboy_data = pd.read_csv('../input/IsisFanboy.csv',encoding = "ISO-8859-1")
about_data.keys()
fanboy_space_split = [str(i).split() for i in fanboy_data['tweets']]

fanboy_handles = [j for i in fanboy_space_split for j in i if '@' in j]

about_space_split = [str(i).split() for i in about_data['tweets']]

about_handles = [j for i in about_space_split for j in i if '@' in j]
print(len(set(fanboy_data['username']))/len(set(fanboy_handles)),

      len(set(about_data['username']))/len(set(about_handles)))
import networkx as nx
fanboy_edges = [(k,j[1:]) for k,i in zip(fanboy_data['username'],fanboy_space_split) for j in i if '@' in j]

about_edges = [(k,j[1:]) for k,i in zip(about_data['username'],about_space_split) for j in i if '@' in j]
about_graph = nx.Graph()

fanboy_graph = nx.Graph()
about_graph.add_edges_from(about_edges)

fanboy_graph.add_edges_from(fanboy_edges)
print(1/(float(fanboy_graph.order())/float(fanboy_graph.size())))

print(1/(float(about_graph.order())/float(about_graph.size())))
fanboy_cc = nx.connected_component_subgraphs(fanboy_graph)

bet_cen = nx.betweenness_centrality([i for i in fanboy_cc][0])
fanboy_cc = nx.connected_component_subgraphs(fanboy_graph)

clo_cen = nx.closeness_centrality([i for i in fanboy_cc][0])
fig, ax = matplotlib.pyplot.subplots()

ax.scatter(list(clo_cen.values()),list(bet_cen.values()))

ax.set_ylim(0.04,0.3)

ax.set_xlim(0.32,0.45)

ax.set_xlabel("Closeness Centrality")

ax.set_ylabel("Betweenness Centrality")

ax.set_yscale('log')

for i, txt in enumerate(list(clo_cen.keys())):

    ax.annotate(txt, (list(clo_cen.values())[i],list(bet_cen.values())[i]))
import re

fanboy_text = [re.sub("[^a-zA-Z]"," ",j).lower() for i in fanboy_space_split for j in i if (not('@' in j) and not('#' in j))]

about_text = [re.sub("[^a-zA-Z]"," ",j).lower() for i in about_space_split for j in i if (not('@' in j) and not('#' in j))]
from sklearn.feature_extraction.text import CountVectorizer

fc_vectorizer = CountVectorizer(stop_words='english',max_features=1000)

fanboy_counts = fc_vectorizer.fit_transform(fanboy_text)

ac_vectorizer = CountVectorizer(stop_words='english',max_features=1000)

about_counts = ac_vectorizer.fit_transform(about_text)
def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        print("Topic #%d:" % topic_idx)

        print(" ".join([feature_names[i]

                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

    print()
from sklearn.decomposition import NMF

n_samples = 2000

n_features = 1000

n_topics = 10

n_top_words = 20

fanboy_nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(fanboy_counts)
fanboy_feature_names = fc_vectorizer.get_feature_names()

print_top_words(fanboy_nmf, fanboy_feature_names, n_top_words)
about_nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(about_counts)
about_feature_names = ac_vectorizer.get_feature_names()

print_top_words(about_nmf, about_feature_names, n_top_words)