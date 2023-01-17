from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split 

import numpy as np
import pandas as pd

from numpy import random as nprand
import random
nprand.seed(100)
random.seed(100)

import networkx as nx

import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams.update({
    'figure.figsize': (15, 15),
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False})
df = pd.read_csv("https://raw.githubusercontent.com/michiru123/HighD-dataset/master/track45classed.csv")
df
Xdf = df.drop('numLaneChanges', 1)
Xdf.head()
y = df['numLaneChanges']
features = np.array(Xdf.columns)
clf = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=500)
clf.fit(Xdf, y)
clf.estimators_[0:3]
clf.estimators_[99].feature_importances_
sum(clf.estimators_[99].feature_importances_ > 0)
def random_forest_to_network(rf_mod, features, thres = 0.1):
    
    G = nx.Graph()
    trees = rf_mod.estimators_
    
    for tree in trees:
        vimp_scores = tree.feature_importances_
        vimps = features[vimp_scores > thres]
        scores = vimp_scores[vimp_scores > thres]
        
        for v,s in zip(vimps,scores):
            try:
                G.nodes[v]['count'] += 1
                G.nodes[v]['score'] += s
            except KeyError:
                G.add_node(v)
                G.nodes[v]['count'] = 1
                G.nodes[v]['score'] = s
            for w in vimps:
                try:
                    G.edges[v, w]['count'] += 1
                except KeyError:
                    G.add_edge(v, w, count=1)
    
    for n,d in G.nodes(data = True):
        G.nodes[n]['avg_vimp'] = d['score']/d['count']
    
    return G
features = np.array(Xdf.columns)

G = random_forest_to_network(clf, features, thres = 0.2)

len(G.nodes)

len(G.edges)
G.nodes(data=True)
G.edges(data =True)
node_sizes = [5000*d['avg_vimp'] for n,d in G.nodes(data = True)]
weights = [np.log2(d['count']) for s, t, d in G.edges(data=True)]

pos=nx.spring_layout(G, k = 0.6)

nx.draw_networkx_nodes(G, pos, alpha=0.8, node_size = node_sizes , node_color = 'green')
nx.draw_networkx_labels(G, pos, font_size = 12, font_color = 'black')
nx.draw_networkx_edges(G, pos, edge_color="green", width = weights, alpha=0.5)