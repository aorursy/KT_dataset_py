import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
nodes = pd.read_csv('../input/stack_network_nodes.csv')
edges = pd.read_csv('../input/stack_network_links.csv')
nodes.head()
edges.head()
G = nx.Graph()
for index, row in nodes.iterrows():
    G.add_node(row["name"],group = row["group"], nodesize = row["nodesize"] )
for index, row in edges.iterrows():
    G.add_edge(row["source"], row["target"], weight = row["value"])
print(nx.info(G))
nx.is_connected(G)
nx.number_connected_components(G)
maximum_connected_component = max(nx.connected_component_subgraphs(G), key=len)
print(nx.__version__)
def draw_graph(G,size):
    nodes = G.nodes()
    color_map = {1:'#f09494', 2:'#eebcbc', 3:'#72bbd0', 4:'#91f0a1', 5:'#629fff', 6:'#bcc2f2',  
             7:'#eebcbc', 8:'#f1f0c0', 9:'#d2ffe7', 10:'#caf3a6', 11:'#ffdf55', 12:'#ef77aa', 
             13:'#d6dcff', 14:'#d2f5f0'}
    node_color= [color_map[d['group']] for n,d in G.nodes(data=True)]
    node_size = [d['nodesize']*10 for n,d in G.nodes(data=True)]
    pos = nx.drawing.spring_layout(G,k=0.70,iterations=60)
    plt.figure(figsize=size)
    nx.draw_networkx(G,pos=pos,node_color=node_color,node_size=node_size,edge_color='#FFDEA2',edge_width=1)
    plt.show()
draw_graph(G,size=(25,25))
cliques = list(nx.find_cliques(G))
clique_number = len(list(cliques))
print(clique_number)
for clique in cliques:
    print(clique)
print(nx.ego_graph(G,'python',radius=2).nodes())
nx.algorithms.clique.cliques_containing_node(G,"python")
nx.algorithms.clique.cliques_containing_node(G,"c++")
nx.algorithms.clique.cliques_containing_node(G,"php")
sorted_cliques = sorted(list(nx.find_cliques(G)),key=len)
max_clique_nodes = set()

for nodelist in sorted_cliques[-4:-1]:
    for node in nodelist:
        max_clique_nodes.add(node)
max_clique = G.subgraph(max_clique_nodes)
print(nx.info(max_clique))
draw_graph(max_clique,size=(10,10))
major_languages = ['c','c++','c#','java','python','ruby','scala','haskell','javascript','sql']
p_language_nodes = []
for language in major_languages:
    neighbors = G.neighbors(language)
    p_language_nodes.extend(neighbors)
programming_language_graph = G.subgraph(set(p_language_nodes))
draw_graph(programming_language_graph,size=(20,20))

plt.hist([node[1] for node in list(G.degree())])
plt.title("Stack Overflow Tag Degree Distribution")
nodes['group'].plot(kind='hist')
nodes['nodesize'].plot(kind="hist")
degree_centrality = nx.degree_centrality(G)

top_10_nodes_by_degree_centrality = sorted(degree_centrality.items(),key=lambda x:x[1],reverse=True)[0:10]

top_10_nodes_by_degree_centrality
betweenness_centrality = nx.betweenness_centrality(G)

top_10_nodes_by_betweenness_centrality = sorted(betweenness_centrality.items(),key=lambda x:x[1],reverse=True)[0:10]
top_10_nodes_by_betweenness_centrality
