# First, import the important packages
import pandas as pd
import networkx as nx
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
# Next read in the edges and create a graph
df = pd.read_csv('../input/group-edges.csv')
g0 = nx.from_pandas_edgelist(df, 
                            source='group1', 
                            target='group2', 
                            edge_attr='weight')

print('The entire MeetUp group graph has {} nodes and {} edges.'.format(
    len(g0.nodes),
    len(g0.edges)))
# Now, let's read in some member metadata and trim to Tech groups
groups = pd.read_csv('../input/meta-groups.csv', index_col='group_id')
tech = groups.loc[groups.category_name == 'Tech']
print('There are {} Tech groups with metadata.'.format(tech.shape[0]))

# Let's trim the graph down to the largest connected Tech network
gt = g0.subgraph(tech.index)
g = [gt.subgraph(c) for c in nx.connected_components(gt)][0]
tech = tech.loc[(n for n in g.nodes)]
print('After trimming, there are {} groups with metadata.\nThese include...'.format(tech.shape[0]))
print(tech.sample(5).group_name.to_string())
plt.figure(dpi=150)

pos = nx.spring_layout(g, k=2)
# pos = nx.random_layout(g)
nx.draw_networkx(g, pos, with_labels=False, node_size=10,
                 width=0.05)

ax = plt.gca()
ax.set_aspect(1)
ax.axis('off')
ax.set_title('Graph Network of Nashville Tech MeetUps')
plt.show()
# Let's run some measures and populate our DataFrame
tech['degree'] = pd.Series(dict(nx.degree(g)))
tech['clustering'] = pd.Series(nx.clustering(g))
tech['centrality'] = pd.Series(nx.betweenness_centrality(g))

# Path length is a little trickier
avg_length_dict = {}
for node, path_lengths in nx.shortest_path_length(g):
    path_lengths = [x for x in path_lengths.values()]
    avg_length_dict[node] = np.mean(path_lengths)
tech['path_length'] = pd.Series(avg_length_dict)
# First, we plot a pairplot to get a sense of the overall relationships
# between each of the measures we derived. On the diagonal is the 
# distribution of values within the variable; off the diagonal are 
# scatterplots showing the correlation between variables.

grid = sns.pairplot(tech[['clustering', 'centrality']], diag_kind='kde')

plt.show()
# Now, let's look at clustering.
fig, ax = plt.subplots(1,1, figsize=(5,10), dpi=100)

gp_order = tech.loc[tech.clustering > 0].sort_values(by='clustering').group_name
sns.barplot(data = tech, x='clustering', y='group_name',
            order = gp_order)

ax.set_title('Average clustering coefficient by Category')
ax.set_yticks(ax.get_yticks()[::4])
ax.set_yticklabels(gp_order[::4], size=16)

plt.show()
tech.sort_values(by='centrality', ascending=False)[['group_name', 'centrality', 'num_members']].head(10)
# Next, let's plot the Number of Members (not degree!) vs. centrality
fig, ax = plt.subplots(1,1, figsize=(10,5))

sns.regplot(data=tech, x='num_members', y='centrality', order=2)
ax.set_title('Centrality vs. Number of Group Members')

plt.show()
print('The ten most "central" groups are...')
print(tech[['group_name', 'num_members', 'clustering', 'centrality']]
          .sort_values(by='centrality', ascending=False)
          .head(10).to_string())
# !pip install python-louvain

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
import community

partition = community.community_louvain.best_partition(g)
tech['community'] = pd.Series(partition)
plt.figure(dpi=150)

pos = community_layout(g, partition)

cdict = {ii: sns.color_palette()[ii] for ii in set(partition.values())}
nx.draw_networkx(g, pos, node_size=60,
                 node_color=[cdict[ii] for ii in partition.values()], 
                 with_labels=False, width=0.15,
                 cmap='rainbow')
plt.axis('off')
plt.show()
for ii in tech.community.unique():
    print('Most central groups in Community {}...'.format(ii))
    tdf = tech.sort_values(by='centrality', ascending=False).loc[tech.community == ii]
    for ix, gp in tdf.head(4).iterrows():
        print('\t{}'.format(gp.group_name, gp.num_members))
gps = ['Nashville PowerShell User Group (NashPUG)', 'Data Science Nashville', 'NashJS', 'Nashville Mobile Developers', 'WordPress Nashville']
tech.loc[tech.group_name.isin(gps), ['group_name', 'num_members', 'clustering', 'centrality', 'community']]