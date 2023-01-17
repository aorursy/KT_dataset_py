import numpy as np
import scipy as sp
import networkx as nx
import pandas as pd
import csv
import matplotlib.pyplot as plt
%matplotlib inline 
node_edge = pd.read_csv('../input/hw3datasets/hw3infmax.csv')     #load node_edge and weight from CSV file
print ("Total Edges = ", len(node_edge))               #Identify lenght of edges size
# Draw Network Graph
G = nx.Graph()
i = 0
while i < len(node_edge):
# Add each edge to Node_edge
    G.add_edge(node_edge.loc[i,'from_node'], node_edge.loc[i,'to_node'], weight = node_edge.loc[i,'weight'])
    i = i+1
    
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

# Set Network Graph Size: [100, 100]
plt.figure(figsize=(50,50))

A = np.array(nx.adjacency_matrix(G).todense())     # Converted Nodegraph to Adjacency Matrix
pos = nx.spring_layout(G)                          # positions for all nodes

nx.draw_networkx_nodes(G, pos, node_size=500, node_color='g')  # Set Nodes Size and Color

nx.draw_networkx_edges(G, pos, edgelist=elarge,
                       width=2)
nx.draw_networkx_edges(G, pos, edgelist=esmall,
                       width=2, alpha=0.5, edge_color='b', style='dashed') # Set Edges Size, Color, Line

nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')    # Set Labels Size

plt.axis('off')
plt.show()

total_nodes = len(G)

print ("total edges =",i)
print ("total nodes =",total_nodes)
# List all Nodes Parameters: Node Degree, Degree Centrality, Clustering , etc.
deg_cen = nx.degree_centrality(G)
print ("node, degree, degree_centrality, clustering")
i = 0
while i < total_nodes:    
    print (i, G.degree(i), deg_cen[i], nx.clustering(G,i))
    i=i+1
# load node_edge and weight from CSV file
initial_nodes = pd.read_csv('../input/hw3datasets/initial_nodes.csv')     
submission_list = initial_nodes
print ("Total Initial nodes =",len(initial_nodes))
# List all Activated Nodes from Initial_node.csv
print ("List of initial activated nodes and cost (degree)")
print ("node, cost")
i = 0
cost = 0
activated_list = []
while i < len(initial_nodes):
    if initial_nodes.loc[i,'activated'] == 1:
        print (initial_nodes.loc[i,'node'], G.degree(i))
        activated_list.append(initial_nodes.loc[i,'node'])
        cost = cost+G.degree(i)
        
    i=i+1
    
print (activated_list)
print ("Number of initial nodes = ",len(activated_list))
print ("Total Cost = ",cost)
iteration_time = 2                  # Iteration round of Influence Maximization
threshold = 0.5                     # Threshold

for x in range(iteration_time):
    i=0
    while i < len(activated_list):
        neighbors_list = [n for n in G[activated_list[i]]]
        print ("neighbors = ",len(neighbors_list))
        j = 0
        while j < len(neighbors_list):
            weight_attr=nx.get_edge_attributes(G,'weight')
            print (activated_list[i],"Activated node found neighbor",neighbors_list[j])
            try:
                if weight_attr[(activated_list[i],neighbors_list[j])] > threshold:
                    print ("ACTIVATED",weight_attr[(activated_list[i],neighbors_list[j])])
                    submission_list.loc[neighbors_list[j],'activated'] = 1         # Activated this node
            except KeyError:
                if weight_attr[(neighbors_list[j],activated_list[i])] > threshold:
                    print ("!ACTIVATED",weight_attr[(neighbors_list[j],activated_list[i])])
                    submission_list.loc[neighbors_list[j],'activated'] = 1         # Activated this node
            
            j=j+1
        i=i+1
    print ("========== END Round ",x+1," ============")
    k = 0
    cost = 0
    activated_list = []
    while k < len(submission_list):
        if submission_list.loc[k,'activated'] == 1:
            print (submission_list.loc[k,'node'], G.degree(k))
            activated_list.append(initial_nodes.loc[k,'node'])
            cost = cost+G.degree(k)
        k=k+1
    
    print (activated_list)
    print ("Round ",x+1,"Activated Nodes",len(activated_list))
    print ("Total Cost = ",cost)
# Write all Activated node result to submission.csv\n
submission_list.to_csv('submission.csv',index=False)
# !!! Ready for Submission !!!
