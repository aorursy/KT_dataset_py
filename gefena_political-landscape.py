import pandas as pd

import itertools

from datetime import timedelta

import os

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline



import seaborn as sns

sns.set_context("poster")

plt.style.use('fivethirtyeight')

#plt.style.use('ggplot')

plt.rcParams['axes.labelweight'] = 'bold'

plt.rcParams['axes.titleweight'] = 'bold'

plt.rcParams['figure.titleweight'] = 'bold' 



from IPython.display import display, HTML



import numpy as np

import math



import datetime

import time

import sys



import networkx as nx





import sklearn

print("sklearn.__version__:",sklearn.__version__)



import pylab as pl

import matplotlib.dates as mdates



print(sys.version)
result_df = pd.read_csv('../input/results_by_booth_2015 - english - v3.csv', encoding='iso-8859-1')

print("Columns:")

print(result_df.columns)

print()

print("df shape:",result_df.shape)

result_df.tail(5)
result_df = result_df.dropna(axis=0, how='any')

result_df = result_df[result_df.votes > 0]

result_df.loc[result_df.Registered_voters == 0,'Registered_voters'] = result_df[result_df.Registered_voters == 0].votes

result_df.shape
block_percent = 0.0325
all_registered_voters = result_df.Registered_voters.sum()

all_votes = result_df.proper_votes.sum()

print("all registerd voters:",all_registered_voters)

print("all_votes:",all_votes)

print("vote percentage:",all_votes/all_registered_voters)

overall_votes_per_party = result_df.iloc[:,8:].sum()

percantage_vote_per_pary = overall_votes_per_party/all_votes

percantage_vote_per_pary = percantage_vote_per_pary[percantage_vote_per_pary.values>block_percent]

percantage_vote_per_pary.sort_values(ascending=False).plot.bar(alpha=0.7,figsize=(16,6))
# Print the large parties

large_parties = percantage_vote_per_pary.index.values

print(large_parties)
non_party_col = list(result_df.iloc[:,0:8].columns)

int_columns = []

int_columns.extend(non_party_col)

int_columns.extend(list(large_parties))

print(int_columns)
res_work_df = result_df.copy()

res_work_df = res_work_df[int_columns]

res_work_df_city = res_work_df.groupby(['settlement_name_english','Settlement_code'])[int_columns[4:]].sum().reset_index()

print(res_work_df_city.shape)

res_work_df_city.head(5)
min_vote_rate = 0.6

min_proper_votes = 300
res_work_df = res_work_df_city.copy()

res_work_df['vote_rate'] = res_work_df.proper_votes / res_work_df.Registered_voters

res_work_df = res_work_df[(res_work_df.vote_rate > min_vote_rate) & (res_work_df.proper_votes > min_proper_votes)]

print(res_work_df.shape)

res_work_df.sample(10)
res_work_df[res_work_df.settlement_name_english.str.contains("BE'ER SHEVA|TEL AVIV|JERU|HAI")] # BE'ER SHEVA / TEL AVIV / JERUSALEM 
res_work_df[res_work_df.vote_rate == np.inf]
res_work_df_percentage_votes = res_work_df.iloc[:,6:-1].div(res_work_df.proper_votes, axis=0)

res_work_df_percentage_votes.head(5)
res_work_df_percentage_votes_transpose = res_work_df_percentage_votes.transpose()

res_work_df_percentage_votes_transpose.head(11)
X = res_work_df_percentage_votes_transpose

X.head(3)
def above_median(fclist):

    med = np.median(fclist)

    return (fclist > med).astype(int)



X = X.apply(above_median, axis=1)
X[1:10]
names = res_work_df_percentage_votes_transpose.index.tolist()
from sklearn.cluster import KMeans



km = KMeans(n_clusters=4, random_state=0).fit(X)

clusters = km.labels_.tolist()

clusters
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2)

results_tsne = tsne.fit(X)



coords = results_tsne.embedding_



colors = ['blue','red','green','cyan','magenta','yellow','black','white']

label_colors = [colors[i] for i in clusters]



plt.figure(figsize=(16,8)) 

plt.subplots_adjust(bottom = 0.1)

plt.scatter(

    coords[:, 0], coords[:, 1], marker = 'o', c=label_colors

    )



for label, x, y in zip(names, coords[:, 0], coords[:, 1]):

    plt.annotate(

        label,

        xy = (x, y), xytext = (-20, 20),

        textcoords = 'offset points', ha = 'right', va = 'bottom',

        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),

        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    

plt.show()
from sklearn.metrics.pairwise import pairwise_distances

from sklearn.preprocessing import MinMaxScaler



x = res_work_df_percentage_votes_transpose



res = pairwise_distances(x, metric='correlation') # cosine / jaccard / correlation / euclidean



distance = pd.DataFrame(res, index=res_work_df_percentage_votes_transpose.index, 

                        columns= res_work_df_percentage_votes_transpose.index)



distance
import scipy

from scipy.cluster import hierarchy



labels = distance.index.values.tolist()

sq_distance = scipy.spatial.distance.squareform(distance)



Z = hierarchy.linkage(sq_distance, 'single')



hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])

fig, axes = plt.subplots(1, 1, figsize=(16, 6))

dn1 = hierarchy.dendrogram(Z, ax=axes, above_threshold_color='y', orientation='top', labels=labels)

plt.show()
new_order_distance = distance.reindex(dn1['ivl'])

new_order_distance = new_order_distance[dn1['ivl']] 
import seaborn as sns

ax = sns.heatmap(new_order_distance)
distance_cutoff = 1

parties = percantage_vote_per_pary.index.tolist()

parties
import itertools

dist_list = list(distance.index)

all_2_org_combos = itertools.combinations(dist_list, 2)

max_dist = distance.max().max()



# Generate graph with nodes: 

G=nx.Graph()



for p in parties:

    G.add_node(p,

               name=p,

               p_vote=float(percantage_vote_per_pary[p]),             

               comm="0") 

    

# Connect nodes:

for combo in all_2_org_combos:

    combo_dist = distance[combo[0]][combo[1]]

    opp_dist = combo_dist - max_dist

    if distance[combo[0]][combo[1]] < distance_cutoff:

        G.add_edge(combo[0],combo[1],weight=float(abs(opp_dist)))

        



n = G.number_of_nodes()

m = G.number_of_edges()     

print("number of nodes in graph G: ",n)

print("number of edges in graph G: ",m)

print()
import community

communities = community.best_partition(G)

mod = community.modularity(communities,G)

print("modularity:", mod)
if m > 0:         

    for k,v in communities.items():

        G.node[k]['comm'] = str(v)

else:

    print("Not runnig Community algorithm because the graph has no edges")
com_values = [communities.get(node) for node in G.nodes()]



p_votes = [d['p_vote'] for n,d in G.nodes(data=True)]

node_size=[v * 3000 for v in p_votes]



plt.figure(figsize=(12,8)) 

pos=nx.fruchterman_reingold_layout(G)

nx.draw_networkx(G,pos, cmap = plt.get_cmap('jet'), node_color = com_values, node_size=node_size, with_labels=True)

plt.show()