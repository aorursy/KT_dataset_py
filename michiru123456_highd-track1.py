# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
df = pd.read_csv("../input/highd-dataset/track1_laneDiff.csv")
df.head()
df.info()
plt.style.use('ggplot')

df.hist(bins=50, figsize=(20,15)) 
df.isnull().sum()
Xdf = df.drop('diff_laneId', 1)
Xdf.head()
y = df['diff_laneId']
features = np.array(Xdf.columns)
clf = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=500)
clf.fit(Xdf, y)
clf.estimators_[0:3]
clf.estimators_[119].feature_importances_
sum(clf.estimators_[119].feature_importances_ > 0)
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

G = random_forest_to_network(clf, features, thres = 0.22)
pr = nx.pagerank(G)
pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)

for i in pr:
	print(i[0], i[1])
#plt.bar(range(len(pr)), pr.values(), align='center')
#plt.xticks(range(len(pr)), list(pr.keys()),rotation=30)

#plt.show()

labels, ys = zip(*pr)
xs = np.arange(len(labels)) 
width = 0.45

plt.bar(xs, ys, width, align='center')

plt.xticks(xs, labels,rotation=30) #Replace default x-ticks with xs, then replace xs with labels
plt.yticks(ys)
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
dmax = max(degree_sequence)

plt.loglog(degree_sequence, "b-", marker="o")
plt.title("Degree rank plot")
plt.ylabel("degree")
plt.xlabel("rank")

# draw graph in inset
plt.axes([0.45, 0.45, 0.45, 0.45])
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
pos = nx.spring_layout(Gcc)
plt.axis("off")
nx.draw_networkx_nodes(Gcc, pos, node_size=20)
nx.draw_networkx_edges(Gcc, pos, alpha=0.4)
plt.show()
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