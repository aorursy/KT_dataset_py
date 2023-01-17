# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
G = []
for i in range (0, 500):
    if (travel_data_final['location'][i] != 'Jilin'):
        G.append((travel_data_final['location'][i], travel_data_final['summary'][i]))
g = nx.Graph()

h = nx.Graph()
#for i in range (0, 100):
 #       h.add_node(G['location'][i])
#for i in range (0, 100):
#        edge = (travel_data_final['location'][i], travel_data_final['summary'][i])
#        h.add_edge(*edge)

lG = len(G)
for i in range (0, lG):
    h.add_edge(*G[i], weight = 1)

num_nodes = len(h)
node_list = list(h.nodes)
for i in range (0, num_nodes):
    if (not nx.has_path(h, target = node_list[i], source = node_list[2])):
        h.remove_node(node_list[i])
        
h.remove_edge('Tokyo','Japan')
h.remove_edge('Tokyo','Chiba Prefecture')
h.remove_edge('Malaysia','Macau')
h.remove_edge('Macau','Taiwan')
h.remove_edge('Beijing','Tianjin')
h.remove_edge('Shaanxi','Xianyang')
h.remove_edge('Hokkaido','Tokyo')
h.remove_edge('Chongqing','Thailand')
h.remove_node('Mudanjiang')
h.remove_node('Chiba')
h.remove_node('Qingdao')
h.remove_node('Hangzhou')
h.remove_node('Heilongjiang')

num_nodes = len(h)
node_list = list(h.nodes)

#my_pos = nx.spring_layout(h, seed = 1)
my_pos = nx.nx_pydot.graphviz_layout(h, prog='twopi', root = 'Wuhan')
my_pos[2] =  np.array([0, 0])
#my_pos = nx.nx_pydot.graphviz_layout(h, prog='neato')

NODES = ['Wuhan']
EDGES = []
LABELS = []

def get_fig(i, r):
    pylab.clf()
    for i in range (0, len(NODES)):
        #node = NODES[i]
        node = 'Wuhan'
        NODES.extend([e for e in (h.neighbors(node))])
        #EDGES.add_edges_from(h.edges(node))
    #g.add_edges_from(G[0:i+1])

    
    #F = nx.compose(h,g)
    
    #NODES = []
    #EDGES = G[0:i]
    #EDGES.append(G[i])
    #NODES.append(EDGES[i][0])
    #NODES.append(EDGES[i][1])
    #LABELS.append(EDGES[i][0])
    #LABELS.append(EDGES[i][1])
    
    #pos = nx.nx_pydot.graphviz_layout(g, prog='neato')
    #nx.draw(g, pos=pos, with_labels=True, node_size= 50, edge_color="red",arrows = False, connectionstyle="rad=0.2")
    nx.draw(h, pos=my_pos, with_labels=False, node_color = "white", node_size= 10, edge_color="white",arrows = False, connectionstyle="rad=0.2")
    nx.draw_networkx_edges(h, pos = my_pos,
                       edgelist=EDGES,
                        edge_color='grey')
    #for j in range (0, i):
    #    NODES.append(EDGES[j][0])
    #    NODES.append(EDGES[j][1])
    #nx.draw_networkx_nodes(h, pos = my_pos, with_labels = True, 
    #                   nodelist=NODES,
    #                   #node_color='r',
    #                   node_size=50,
    #                   alpha=0.8)
    hub_ego = nx.ego_graph(h, 'Wuhan', radius = r)
    #pos = nx.circular_layout(hub_ego)
    #pos = nx.nx_pydot.graphviz_layout(h, prog='twopi')
    nx.draw(hub_ego, pos = my_pos,edge_color = 'grey', node_size=50, with_labels=False)
    #nx.draw_networkx_labels(h, pos=my_pos, font_size = 8)
    
    
    
    

num_plots = 46
j = 0
pylab.show()

#for i in range(num_plots):
#for i in range(0, 99):
    #print(chr(27) + "[2J")
for r in range (0, 3):
    i = 100
    get_fig(i, r)
    pylab.draw()
    pylab.savefig('h' + str(r) + '.png', dpi = 300)
    #pause(0.01)
    #j = j+1