# First, import the important packages
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

# Next read in the edges and create a graph
df = pd.read_csv('../input/group-edges.csv')
g = nx.from_pandas_edgelist(df, 
                            source='group1', 
                            target='group2', 
                            edge_attr='weight')

print('The member graph has {} nodes and {} edges.'.format(len(g.nodes),
                                                          len(g.edges)))
pos = nx.spring_layout(g)
nx.draw_networkx(g, pos)
# Circular Layout
fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=150)

pos = nx.circular_layout(g)
nx.draw_networkx_nodes(g, pos, node_size=10, 
                       node_color='xkcd:muted blue')
nx.draw_networkx_edges(g, pos, alpha=0.05)

ax.axis('off')
plt.show()
# Random Layout
fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=150)

pos = nx.random_layout(g)
nx.draw_networkx_nodes(g, pos, node_size=50,
                      node_color='xkcd:muted green')
nx.draw_networkx_edges(g, pos, alpha=0.05)

ax.axis('off')
plt.show()
# Spring Layout
fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=150)

pos = nx.spring_layout(g, k=2)
nx.draw_networkx_nodes(g, pos, node_size=50,
                      node_color='xkcd:muted purple')
nx.draw_networkx_edges(g, pos, alpha=0.03)

ax.axis('off')
plt.show()
# Weight nodes by degree and edge size by width 
fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=150)

pos = nx.spring_layout(g, k=2)
node_sizes = [g.degree[u] for u in g.nodes]
nx.draw_networkx_nodes(g, pos, node_size=node_sizes,
                       node_color='xkcd:muted purple')

edge_widths = [d['weight'] for u,v,d in g.edges(data=True)]
nx.draw_networkx_edges(g, pos, width=edge_widths, alpha=0.03)

ax.axis('off')
plt.show()
