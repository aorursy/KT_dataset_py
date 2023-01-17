import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import networkx as nx

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors



import pandas as pd



pd.set_option('display.expand_frame_repr', False)



'''

Connectome.csv

Neurons_to_Muscles.csv

Sensory.csv

spatialpositions

'''



G = nx.DiGraph()



from matplotlib.pyplot import figure



df = pd.read_csv('../input/celeganstp-master/CElegansTP-master/data/Connectome.csv')

print(df.head())



neurons = df['Neuron'].unique() 



print(df['Neurotransmitter'].unique())

for value in neurons:

    G.add_node(value)

for index, row in df.iterrows():

    color = 'red' if row['Neurotransmitter'] == 'exc' else 'blue'

    G.add_edge(row['Neuron'], row['Target'] , color=color )



plt.figure(figsize=(50,50))

nx.draw(G, with_labels=True)

plt.show()
#from networkx.algorithms.centrality import degree_centrality

# Centrality

#print(degree_centrality(G))



def draw(G, measures, measure_name):

    plt.figure(figsize=(40,40))

    #spring_layout

    pos = nx.random_layout(G)

    #spectral_layout

    #nx.draw(G, with_labels=True, font_weight='bold')

    nodes = nx.draw_networkx_nodes(G,with_labels=True, pos=pos, node_size=300, cmap=plt.cm.plasma, 

                                   node_color=list(measures.values()),

                                   nodelist=list(measures.keys()))

    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))

    

    # labels = nx.draw_networkx_labels(G, pos)

    edges = G.edges()

    #weights = [G[u][v]['weight'] for u,v in edges]

    #, width=weights

    colors = [G[u][v]['color'] for u,v in edges]

    edges = nx.draw_networkx_edges(G, pos=pos , edge_color = colors , alpha = 1 )

    labels= nx.draw_networkx_labels(G, pos=pos , font_color = 'white' , font_size = 6 )



    plt.title(measure_name)

    plt.colorbar(nodes)

    plt.axis('off')

    plt.show()



draw(G, nx.in_degree_centrality(G), 'DiGraph Degree Centrality')



# Modularity

'''

from networkx.algorithms.community import greedy_modularity_communities

c = list(greedy_modularity_communities(G))

sorted(c[0])

'''