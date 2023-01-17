import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.kclique import k_clique_communities
import matplotlib.pyplot as plt
!ls
tG = nx.read_edgelist('../input/tedges.txt')
communities = list(k_clique_communities(tG, 3))
%matplotlib inline
pos = nx.spring_layout(tG)
colors = ["violet", "black", "orange", "cyan", "blue", "green", "yellow", "indigo", "pink", "red"]
for i in range(len(communities)):
    graph = communities[i]
    node_list = [node for node in graph]
    nx.draw(tG, pos, nodelist=node_list, node_color=colors[i%10], node_size=50, alpha=0.8)
# From SO: https://stackoverflow.com/questions/40941264/how-to-draw-a-small-graph-with-community-structure-in-networkx
def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos
dict_communities = {}

for i, c in enumerate(communities):
    for node in c:
        dict_communities[node] = i + 1
        
for node in tG:
    if node not in dict_communities.keys():
        dict_communities[node] = -1
pos = community_layout(tG, dict_communities)
from matplotlib import cm
colors = []
for node in tG.nodes:
    colors.append(cm.Set1(dict_communities[node]))
plt.figure(figsize=(20,20))
nx.draw_networkx_nodes(tG, pos, node_color=colors, node_size=20)
nx.draw_networkx_edges(tG, pos, alpha=0.05)
plt.axis('off')
plt.show()
from networkx import edge_betweenness_centrality
from random import random

def most_valuable_edge(G):
    centrality = edge_betweenness_centrality(G)
    max_cent = max(centrality.values())
    # Scale the centrality values so they are between 0 and 1,
    # and add some random noise.
    centrality = {e: c / max_cent for e, c in centrality.items()}
    # Add some random noise.
    centrality = {e: c + random() for e, c in centrality.items()}
    return max(centrality, key=centrality.get)
gn_generator = girvan_newman(tG, most_valuable_edge)
from itertools import islice
gn_communities = next(islice(gn_generator, 3, None)) # Do 3 iterations only
type(gn_communities)
gn_dict_communities = {}

for i, c in enumerate(gn_communities):
    print ("Community {}".format(i))
    for node in c:
        gn_dict_communities[node] = i + 1
        
for node in tG:
    if node not in gn_dict_communities.keys():
        gn_dict_communities[node] = -1
gn_pos = community_layout(tG, gn_dict_communities)
from matplotlib import cm
gn_colors = []
for node in tG.nodes:
    gn_colors.append(cm.Set1(gn_dict_communities[node]))
plt.figure(figsize=(20,20))
nx.draw_networkx_nodes(tG, gn_pos, node_color=gn_colors, node_size=20)
nx.draw_networkx_edges(tG, gn_pos, alpha=0.05)
plt.axis('off')
plt.show()
import pandas as pd
fsq = pd.read_csv('../input/fedges.txt', delim_whitespace=True)
fsq.columns = ['source', 'dest']

mapper = pd.read_csv('../input/twitter_foursquare_mapper.dat.txt')

fsq_set = set(mapper['foursquare'])
clean = fsq[(fsq['source'].isin(fsq_set)) & (fsq['dest'].isin(fsq_set))]

fG = nx.from_pandas_edgelist(df=clean, source='source', target='dest')
len(fG.edges)
len(fG.nodes)
# Mapping twitter names to foursqare IDs
fG = nx.relabel_nodes(fG, pd.Series(mapper.twitter_username.values, index=mapper.foursquare).to_dict())
components = list(nx.connected_component_subgraphs(fG))
len(components)
# fsq_communities = list(k_clique_communities(components[0], 3))
# not enough memory
fsq_gn_generator = girvan_newman(components[0])
from itertools import islice
fsq_gn_communities = next(islice(fsq_gn_generator, 3, None)) # Do 3 iterations only
# fsq_gn_communities = (next(gn_generator))
type(fsq_gn_communities)
fsq_gn_dict_communities = {}

for i, c in enumerate(fsq_gn_communities):
#     print ("Community {}".format(i + 1))
    for node in c:
        fsq_gn_dict_communities[node] = (i + 1)

for component in components:
    if component == components[0]:
        for node in component:
            if node not in fsq_gn_dict_communities.keys():
                fsq_gn_dict_communities[node] = -1
    else:
        val = int((max(fsq_gn_dict_communities.values()) + random() * 10)) % 8
#         print ("Component color = {}\t {}".format(val, cm.Set1(val)))
        for node in component:
            fsq_gn_dict_communities[node] = val
fsq_gn_pos_list = list()
for component in components:
    fsq_gn_pos_list.append(community_layout(component, fsq_gn_dict_communities))
from matplotlib import cm

fsq_gn_colors = []
for node in fG.nodes:
    val = fsq_gn_dict_communities[node]%8
    color = cm.Set1(fsq_gn_dict_communities[node]%8)
    fsq_gn_colors.append(color)
#     print("Color = {}\t{}".format(val, color))
plt.figure(figsize=(20,20))
for item, component in zip(fsq_gn_pos_list, components):
    nx.draw_networkx_nodes(component, item, node_color=fsq_gn_colors, node_size=20)
    nx.draw_networkx_edges(component, item, alpha=0.05)
plt.axis('off')
plt.show()
print("Number of communities detected after 3 iterations of Girvan–Newman:",
     "\nTwitter: {}\nFoursqare: {}\n".format(len(gn_communities), 
                                             len(fsq_gn_communities)))
print ("With sizes\nTwitter\tFoursqare")
for t, f in zip(gn_communities, fsq_gn_communities):
    print("{}\t{}".format(len(t), len(f)))
from sklearn.metrics import jaccard_similarity_score
def jaccard_distance(set1, set2):
    intersection = len(set1.intersection(set2))
    union = (len(set1) + len(set2)) - intersection
    return float(intersection / union)
import itertools
jaccard = []
c = list(itertools.product(list(gn_communities), list(fsq_gn_communities)))

# print(jaccard_similarity_score(list(gn_communities), list(fsq_gn_communities)))

for comb in c:
#     print(type(comb[0]))
    jaccard.append(jaccard_distance(comb[0], comb[1]))
import seaborn as sns
sns.set()
sns.distplot(jaccard)
fsq_gn_communities
