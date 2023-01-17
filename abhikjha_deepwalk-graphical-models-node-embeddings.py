# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import networkx as nx

import pandas as pd

import numpy as np

import random

from tqdm import tqdm

from sklearn.decomposition import PCA



import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/seealsology-life-insurance-dataset/seealsology-data.tsv', sep = "\t")

df.head()
G = nx.from_pandas_edgelist(df, "source", "target", edge_attr=True, create_using=nx.Graph())
len(G)
def get_randomwalk(node, path_length):

    

    random_walk = [node]

    

    for i in range(path_length-1):

        temp = list(G.neighbors(node))

        temp = list(set(temp) - set(random_walk))    

        if len(temp) == 0:

            break



        random_node = random.choice(temp)

        random_walk.append(random_node)

        node = random_node

        

    return random_walk
get_randomwalk('pension', 10)
# get list of all nodes from the graph

all_nodes = list(G.nodes())



random_walks = []

for n in tqdm(all_nodes):

    for i in range(5):

        random_walks.append(get_randomwalk(n,10))

        

# count of sequences

len(random_walks)
from gensim.models import Word2Vec



import warnings

warnings.filterwarnings('ignore')
# train skip-gram (word2vec) model

model = Word2Vec(window = 4, sg = 1, hs = 0,

                 negative = 10, # for negative sampling

                 alpha=0.03, min_alpha=0.0007,

                 seed = 14)



model.build_vocab(random_walks, progress_per=2)



model.train(random_walks, total_examples = model.corpus_count, epochs=20, report_delay=1)
model.similar_by_word('critical illness insurance')
G_new =nx.from_pandas_edgelist(df[df['source']=="critical illness insurance"],  "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())
plt.figure(figsize=(12,12))

pos = nx.spring_layout(G_new, k = 0.1) # k regulates the distance between nodes

nx.draw(G_new, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
G_new =nx.from_pandas_edgelist(df[df['source']=="life insurance"],  "target", 

                          edge_attr=True, create_using=nx.MultiDiGraph())
plt.figure(figsize=(12,12))

pos = nx.spring_layout(G_new, k = 0.1) # k regulates the distance between nodes

nx.draw(G_new, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)

plt.show()
X = model[model.wv.vocab]
pca = PCA(n_components=2)

result = pca.fit_transform(X)

# create a scatter plot of the projection

plt.figure(figsize=(15,15))

plt.scatter(result[:, 0], result[:, 1])

words = list(model.wv.vocab)

for i, word in enumerate(words):

    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()
X = model[model.wv.index2entity[:50]]
pca = PCA(n_components=2)

result = pca.fit_transform(X)

# create a scatter plot of the projection

plt.figure(figsize=(15,15))

plt.scatter(result[:, 0], result[:, 1])

words = list(model.wv.index2entity[:50])

for i, word in enumerate(words):

    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()