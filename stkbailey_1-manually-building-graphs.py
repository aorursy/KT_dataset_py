import networkx as nx

g = nx.Graph()
g
# Add nodes
g.add_node('A')
g.add_node('B')
g.add_node('C')
g.add_node('D')

print(g.nodes)
# Add edges
g.add_edge(u='A', v='B')
g.add_edge('A', 'D')
g.add_edge('B', 'D')
g.add_edge('B', 'C')
g.add_edge('C', 'D')

# Or, you could use:
# g.add_edges_from([('A','B'), ('A', 'D'), ...])

print(g.edges)
# Add edge weights
edge_weights = {('A', 'B'): 1, ('A', 'D'): 2, 
                ('B', 'D'): 3, ('B', 'C'): 4 , 
                ('C', 'D'): 5} 
nx.set_edge_attributes(g, edge_weights, 'weight')

print(g.edges(data=True))
g['A']
g['A']['B']
g['A']['B']['weight']
# You can also use dict/list-comps
[g[u][v]['weight'] for u,v in g.edges]
# Simplest possible graph
pos = nx.circular_layout(g)
nx.draw_networkx(g, pos)
# Graph plotting edges separately and by weight
pos = nx.random_layout(g)
nx.draw_networkx_nodes(g, pos)

edge_widths = [d['weight'] for u,v,d in g.edges(data=True)]
nx.draw_networkx_edges(g, pos, width=edge_widths)
edge_list = [(x,y) for x in 'ABCDEFKL' for y in 'DAHLOUVW']
g2 = nx.from_edgelist(edge_list)

print('Nodes: ', g2.nodes)
print('Edges: ', g2.edges)
pos = nx.circular_layout(g2)
nx.draw_networkx(g2, pos)