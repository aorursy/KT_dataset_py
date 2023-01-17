import networkx as nx

# Creating a Graph 
G = nx.Graph() # Right now G is empty

# Add a node
G.add_node(1) 
G.add_nodes_from([2,3]) # You can also add a list of nodes by passing a list argument

# Add edges 
G.add_edge(1,2)

e = (2,3)
G.add_edge(*e) # * unpacks the tuple
G.add_edges_from([(1,2), (1,3)]) # Just like nodes we can add edges from a list
G.nodes()
#NodeView((1, 2, 3))

G.edges()
#EdgeView([(1, 2), (1, 3), (2, 3)])

G[1] # same as G.adj[1]
#AtlasView({2: {}, 3: {}})

G[1][2]
G.edges[1, 2]
%matplotlib inline
import matplotlib.pyplot as plt
nx.draw(G)
# source: https://www.analyticsvidhya.com/blog/2018/04/introduction-to-graph-theory-network-analysis-python-codes/
