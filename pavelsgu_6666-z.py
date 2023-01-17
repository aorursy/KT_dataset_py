import networkx as nx
G = nx.Graph()
G.add_edge('A','B');
G.add_edge('A','C');
G.add_edge('B','D');
G.add_edge('B','E');
G.add_edge('D','E');
nx.draw_networkx(G)
import networkx as nx
import pandas as pd
import numpy as np

fb=nx.read_edgelist('/kaggle/input/facebook-social-network/facebook-combined.txt')
fb_n, fb_k = fb.order(), fb.size()
fb_avg_deg = fb_k / fb_n
print ('Nodes:', fb_n)
print ('Edges:', fb_k)
print ('Average degree:', fb_avg_deg)
import math
degrees = fb.degree().values()
degree_hist = plt.hist(degrees , 100)
fb_prun = nx.read_edgelist('/kaggle/input/facebook-social-network/facebook-combined.txt')
fb_prun. remove_node('0')
print ('Remaining nodes:', fb_prun .number_of_nodes())
print('New # connected components:', nx.number_connected_components(fb_prun))
import networkx as nx
fb = nx.read_edgelist('/kaggle/input/facebook-social-network/facebook-combined.txt')
fb_components = nx.connected_components(fb_prun)
print("Size of the connected components", [len(c) for c in fb_components])
import networkx as nx
import pandas as pd
fb = nx.read_edgelist('/kaggle/input/facebook-social-network/facebook-combined.txt')
fb_prun = nx.read_edgelist('/kaggle/input/facebook-social-network/facebook-combined.txt')

degree_cent_fb = nx.degree_centrality(fb)
print('Facebook degree centrality:',
      sorted (degree_cent_fb.items(), 
              key=lambda x:x[1],
              reverse = True)[:10])
degree_hist = plt.hist(list (degree_cent_fb. values()), 100)
plt.loglog (degree_hist [1][1:],
            degree_hist[0], 'b', marker = '0')
import networkx as nx
import pandas as pd
fb = nx.read_edgelist('/kaggle/input/facebook-social-network/facebook-combined.txt')
fb_prun = nx.read_edgelist('/kaggle/input/facebook-social-network/facebook-combined.txt')

betweenness_fb = nx. betweenness_centrality(fb)
closeness_fb = nx. closeness_centrality(fb)
eigencentrality_fb = nx. eigenvector_centrality(fb)
print ("Facebook betweenness centrality:",  sorted (betweenness_fb.items(), key = lambda x: x[1],reverse = True)[:10])
print ("Facebook closeness centrality:", sorted (closeness_fb.items(),  key = lambda x: x[1], everse = True)[:10])
print ("Facebook eigenvector centrality:", sorted (eigenvector_fb.items(),  key = lambda x: x[1],reverse = True)[:10])
def trim_degree_centrality(graph, degree=0.01):
    g=graph.copy()
    d=nx.degree_centrality(g)
    for n in g.nodes():
        if d[n] <= degree:
            g.remove_node(n)
    return (g)   
thr = 21.0/(fb.order() - 1.0)
print ("Degree centrality threshold:", thr)
fb_trimmed = trim_degree_centrality(fb , degree = thr)
print ("Remaning # nodes:", len(fb_trimmed))
