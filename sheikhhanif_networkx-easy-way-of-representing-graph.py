import networkx as nx
import matplotlib.pyplot as plt

# Creating an empty graph
G = nx.Graph()

# adding single node in graph
G.add_node("A")

# to add multiple nodes in graph
G.add_nodes_from(["B", "C", "D", "E", "F"])

# to display the nodes
G.nodes()
# to add edge between nodes
G.add_edge("A","B")

# to add multiple edges at a time
G.add_edges_from([("A","C"), ("A","D"), ("A","E"), ("E","F")])

# to view the edges
G.edges()
# to know the number of edges
print(G.number_of_edges())

# to know the number of nodes
print(G.number_of_nodes())
# We have created a graph with 6 nodes and 5 edges.
# Now let's view our graph.

nx.draw(G, with_labels=True, node_color="red", edge_color="blue")
# to remove node from graph
G.remove_node("F")

# to remove edge from graph
G.remove_edge("A","E")
# We can now visualize our graph after removing
# node F and edge A to E
nx.draw(G, with_labels=True, node_color="red", edge_color="blue")
from scipy.stats import bernoulli
def er_graph(N, p):
    """ Generate an ER-Graph"""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p=p):
                G.add_edge(node1, node2)
    return G
# lets call our er_graph model and visualize
my_er_graph = er_graph(40, 0.2)
nx.draw(my_er_graph, node_color="blue", edge_color="red")
# if we increment the probalility the graph will be densly connected
nx.draw(er_graph(40, 0.5), node_color="blue", edge_color="red")
