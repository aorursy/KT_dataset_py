!pip install -U node2vec
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import json
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from node2vec import Node2Vec

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import re

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
list_files = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename[-10:] == 'Tweets.CSV':
            df = pd.read_csv(os.path.join(dirname, filename), index_col=None, header=0)
            list_files.append(df)

df = pd.concat(list_files, axis=0, ignore_index=True)
df.shape
df.head()
df.columns.tolist()
tweets_by_country = df.groupby('is_quote').country_code.value_counts().unstack(0).reset_index()
tweets_by_country = tweets_by_country.sort_values(by=False, ascending=False)
tweets_by_country.columns = tweets_by_country.columns.astype(str)
tweets_by_country['Total'] = tweets_by_country['False'] + tweets_by_country['True']
tweets_by_country['PCT_False'] = tweets_by_country['False'] / tweets_by_country['Total']
tweets_by_country['PCT_True'] = tweets_by_country['True'] / tweets_by_country['Total']
# cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

# plt.figure(figsize=(10,10))
# ax = sns.scatterplot(x='PCT_False', y='Total', data=tweets_by_country, alpha = 0.5, s = tweets_by_country['Total'])

# for line in range(0,tweets_by_country.shape[0]):
#      ax.text(tweets_by_country['PCT_False'][line], tweets_by_country['Total'][line], tweets_by_country['country_code'][line], 
#              horizontalalignment='left', size='small', color='black')
f, ax = plt.subplots(figsize=(14, 5))

sns.set_color_codes("pastel")
sns.barplot(x="country_code", y="Total", data=tweets_by_country[:20],
            label="Total", color="b")


sns.set_color_codes("muted")
sns.barplot(x="country_code", y="True", data=tweets_by_country[:20],
            label="Quoted", color="b")

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Tweets quoted and not quoted by Country")
sns.despine(left=True, bottom=True)
# ax = pd.to_datetime(df['account_created_at']).map(lambda x: datetime.date(x)).value_counts().plot(figsize=(14,5))
# ax.set_xlim('2010-01-01','2020-03-30')
# ax.set_title('Number of accounts created by day')
# ax.set_xlabel('Day')
df['hour'] = pd.to_datetime(df['created_at']).map(lambda x: x.hour)
ax = df['hour'].value_counts().sort_index().plot(figsize=(14,5), marker='o')
df[df['is_quote']==True]['hour'].value_counts().sort_index().plot(figsize=(14,5), marker='o', ax=ax)
df[df['is_quote']==False]['hour'].value_counts().sort_index().plot(figsize=(14,5), marker='o', ax=ax)
ax.set_title('Number of tweets by hour')
ax.set_xlabel('Hour')

df.groupby('is_quote').hour.value_counts(normalize=True).unstack(0).plot(figsize=(14,5))
df.groupby('hour').is_quote.value_counts(normalize=True).unstack(1)[True].plot(figsize=(14,5))
df['length_text'] = df['text'].map(lambda x: len(x))
top_countries = tweets_by_country[:5]['country_code'].tolist()
# Sort the dataframe by target
# Use a list comprehension to create list of sliced dataframes
targets = [df.loc[df['country_code'] == val] for val in top_countries]

# Iterate through list and plot the sliced dataframe

f, ax = plt.subplots(figsize=(14, 6))

for target in targets:
    sns.distplot(target[['length_text']], hist=True, rug=False, 
                 kde=False, hist_kws=dict(alpha=0.1), label=target['country_code'].unique())

ax.legend(ncol=1, loc="upper right", frameon=True)
ax.set(xlim=(0, 400), ylabel="",
       xlabel="Distribution of length of tweets")
sns.despine(right=True, top=True)
plt.show()
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

plt.figure(figsize=(10,10))
ax = sns.scatterplot(x='retweet_count', y='favourites_count', data=df, alpha = 0.5)

for line in range(0,tweets_by_country.shape[0]):
     ax.text(df['retweet_count'][line], df['favourites_count'][line], df['country_code'][line], 
             horizontalalignment='left', size='small', color='black')
def extract(start, tweet):

    words = tweet.split()
    return [word[1:] for word in words if word[0] == start]

def strip_punctuation(s):
    return s.translate(str.maketrans('','','!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))

def extract_hashtags(tweet):
    hashtags = [strip_punctuation(tag) for tag in extract('#', tweet)]

    result = []
    for tag in hashtags:
        if tag.lower() not in result:  
            result.append(tag.lower())
    return result
df['hashtags'] = df['text'].apply(extract_hashtags)
df2 = df[['text', 'hashtags', 'country_code']]
df2 = df2[[len(p)>1 for p in df2['hashtags']]]
df2.head()
country='US'

list_Hashtags = df2[df2.country_code==country]['hashtags'].tolist()
                   
H = nx.DiGraph()

for L in list_Hashtags:
    for i in range(len(L)):
        for j in range(i,len(L)):
            H.add_edge(L[i], L[j])
print('Number of nodes: {}'.format(H.number_of_nodes()))
print('Number of edges: {}'.format(H.number_of_edges()))
def get_strongly_cc(G, node):
    """ get storngly connected component of node""" 
    for cc in nx.strongly_connected_components(G):
        if node in cc:
            return cc
    else:
        return set()

def get_weakly_cc(G, node):
    """ get weakly connected component of node""" 
    for cc in nx.weakly_connected_components(G):
        if node in cc:
            return cc
    else:
        return set()
    

def connected_component_subgraphs(G):
    """ get all connected component of node""" 
    for c in nx.connected_components(G):
        yield G.subgraph(c)
        
    
def get_strongly_gcc(G):
    """ get the giant strongly connected component of G""" 
    SGcc = []
    for node in G.nodes():
        strong_component = get_strongly_cc(G, node)  
        if len(strong_component) > len(SGcc):
            SGcc = strong_component
    return SGcc

def get_weakly_gcc(G):
    """ get the giant weakly connected component of G""" 
    WGcc = []
    for node in G.nodes():
        strong_component = get_weakly_cc(G, node)  
        if len(strong_component) > len(WGcc):
            WGcc = strong_component
    return WGcc
SGH1 = get_strongly_gcc(H)
SGH1 = H.subgraph(SGH1)
degrees_h = H.degree()

nodes_highest_degree = [n for (n, deg) in degrees_h if degrees_h[n] > 10]
H_highest = H.subgraph(nodes_highest_degree)

degrees_highest = H_highest.degree()
print('Number of nodes: {}'.format(SGH1.number_of_nodes()))
print('Number of edges: {}'.format(SGH1.number_of_edges()))
SGH2 = sorted(connected_component_subgraphs(H), key=len, reverse=True)[0]

print('Number of nodes: {}'.format(SGH2.number_of_nodes()))
print('Number of edges: {}'.format(SGH2.number_of_edges()))
degrees_h = SGH1.degree()
with plt.style.context('ggplot'):
    
    plt.loglog(sorted([n[1] for n in list(degrees_h)], reverse=True))
    plt.title("Degree rank plot")
    plt.ylabel("degree")
    plt.xlabel("rank")
plt.figure(num=None, figsize=(15, 15), dpi=50, facecolor='w', edgecolor='k')

pos = nx.spring_layout(H_highest)

# nodes
nx.draw_networkx_nodes(H_highest, pos, nodelist=dict(degrees_highest).keys(), 
                       node_size=[v * 3 for v in dict(degrees_highest).values()], alpha=0.5)

# edges
nx.draw_networkx_edges(H_highest, pos, width=0.3, alpha=0.3, edge_color='b')

# labels
nx.draw_networkx_labels(H_highest, pos, font_size=7, font_family='sans-serif')

plt.axis('off')
plt.show()
nodes = df[['user_id', 'country_code']].drop_duplicates().dropna()
edges = df[~df['reply_to_user_id'].isna()][['user_id', 'reply_to_user_id']].drop_duplicates()
nodes = pd.merge(nodes, edges.groupby('user_id').count().rename(columns={'reply_to_user_id': 'out'}), how='left',
            left_on='user_id', right_on='user_id').fillna(0)

nodes = pd.merge(nodes, edges.groupby('reply_to_user_id').count().rename(columns={'user_id': 'in'}), how='left',
            left_on='user_id', right_on='reply_to_user_id').fillna(0)

nodes = nodes[nodes['in'] > 0]
nodes = nodes[nodes['out'] > 0]
nodes
G = nx.from_pandas_edgelist(edges, 'user_id', 'reply_to_user_id', create_using=nx.DiGraph())
nx.set_node_attributes(G, pd.Series(nodes['in'].to_list(), index=nodes.user_id).to_dict(), 'in')
nx.set_node_attributes(G, pd.Series(nodes['out'].to_list(), index=nodes.user_id).to_dict(), 'out')
nx.set_node_attributes(G, pd.Series(nodes['country_code'].to_list(), index=nodes.user_id).to_dict(), 'country')
print('Number of nodes: {}'.format(G.number_of_nodes()))
print('Number of edges: {}'.format(G.number_of_edges()))
nodes_us = [x for x,y in G.nodes(data=True) if ('country' in y.keys() and 'US' in y['country'])]
G_US = G.subgraph(nodes_us)
degrees_us = G_US.degree()
plt.figure(num=None, figsize=(15, 15), dpi=50, facecolor='w', edgecolor='k')

pos = nx.spring_layout(G_US)

# nodes
nx.draw_networkx_nodes(G_US, pos, nodelist=dict(degrees_us).keys(), 
                       node_size=[v * 40 for v in dict(degrees_us).values()], alpha=0.5)

# edges
nx.draw_networkx_edges(G_US, pos, width=0.3, alpha=1, edge_color='b')

plt.axis('off')
plt.show()
print('Number of nodes: {}'.format(G_US.number_of_nodes()))
print('Number of edges: {}'.format(G_US.number_of_edges()))
n2v_obj = Node2Vec(H_highest, dimensions=10, walk_length=5, num_walks=10, p=1, q=1, workers=1)
#node2vec = Node2Vec(H, dimensions=64, walk_length=30, num_walks=200, workers=4) 
n2v_model = n2v_obj.fit(window=3, min_count=1, batch_words=4)
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_embeddings(model, nodes):
    """Extract representations from the node2vec model"""
    embeddings = [list(model.wv.get_vector(n)) for n in nodes]
    embeddings = np.array(embeddings)
    print(embeddings.shape)
    return embeddings

def dim_reduction(embeddings, labels, frac=None, tsne_obj=TSNE(n_components=2)):
    """Dimensionality reduction with t-SNE. Sampling random instances is supported."""
    N = len(embeddings)
    print(N)
    if frac != None:
        idx = np.random.randint(N, size=int(N*frac))
        X = embeddings[idx,:]
        X_labels = [labels[i] for i in idx]
    else:
        X = embeddings
        X_labels = labels
    X_embedded = tsne_obj.fit_transform(X)
    print("t-SNE object was trained on %i records!" % X.shape[0])
    print(X_embedded.shape)
    return X_embedded, X_labels

def visu_embeddings(X_embedded, X_labels=None, colors = ['r','b']):
    if X_labels != None:
        label_map = {}
        for i, l in enumerate(usr_tsne_lab):
            if not l in label_map:
                label_map[l] = []
            label_map[l].append(i)
        fig, ax = plt.subplots(figsize=(15,15))
        for i, lab in enumerate(label_map.keys()):
            print(lab)
            idx = label_map[lab]
            x = list(X_embedded[idx,0])
            y = list(X_embedded[idx,1])
            #print(len(x),len(y))
            ax.scatter(x, y, c=colors[i], label=lab, alpha=0.5, edgecolors='none')
        plt.legend()
    else:
        plt.figure(figsize=(15,15))
        x = list(X_embedded[:,0])
        y = list(X_embedded[:,1])
        plt.scatter(x, y, alpha=0.5)
for node, _ in n2v_model.most_similar('coronavirus'):
    # Show only players
    if len(node) > 3:
        print(node)
embeddings = [list(n2v_model.wv.get_vector(n)) for n in H_highest.nodes]
embeddings = np.array(embeddings)
print(embeddings.shape)
embeddings =get_embeddings(n2v_model, H_highest.nodes)
H_highest.nodes()
plt.figure(figsize=(15,15))
x = list(embeddings[:,0])
y = list(embeddings[:,1])

fig, ax = plt.subplots(figsize=(15,15))
ax.scatter(x, y, alpha=0.5)

for i, txt in enumerate(list(Highest.nodes())):
     if y[i] < -0.3*x[i] -0.1 or  y[i] > -0.3*x[i] + 0.3:
        ax.annotate(txt, (x[i], y[i]))
    
plt.show()
