!apt-get install -y graphviz libgraphviz-dev pkg-config
!pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"
!pip install graphviz
!pip install pgmpy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pgmpy
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.readwrite import BIFReader
from networkx.drawing.nx_agraph import graphviz_layout
reader = BIFReader("../input/asia.bif")
asia = reader.get_model()
nx.draw(asia, with_labels=True, pos=graphviz_layout(asia))
plt.show()
for i in asia.predecessors("either"):
  print (i)
for i in asia.successors("either"):
  print (i)
source = 'dysp'
observed = ['either']  
from collections import defaultdict

to_be_visited = set()    
visited = defaultdict(lambda: False)
top_marked = defaultdict(lambda: False)
bottom_marked = defaultdict(lambda: False)

to_be_visited.add((source,"child"))

while (len(to_be_visited)>0):
    current_node = to_be_visited.pop()
    node_name = current_node[0]
    came_from = current_node[1]
    visited[node_name] = True
    if((node_name not in observed) and (came_from=='child')):
        if(top_marked[node_name]==False):
            top_marked[node_name]=True
            for parent in asia.predecessors(node_name):
                to_be_visited.add((parent,"child"))
        if(bottom_marked[node_name]==False):
            bottom_marked[node_name]=True
            for child in asia.successors(node_name):
                to_be_visited.add((child,"parent"))
    if(came_from=='parent'):
        if((node_name in observed) and (top_marked[node_name]==False)):
            top_marked[node_name] = True
            for parent in asia.predecessors(node_name):
                to_be_visited.add((parent,"child"))
        if((node_name not in observed) and (bottom_marked[node_name]==False)):
            bottom_marked[node_name]=True
            for child in asia.successors(node_name):
                to_be_visited.add((child,"parent"))

for node_name in asia:
    if (bottom_marked[node_name]==False and node_name not in observed):
        print (node_name)
asia.active_trail_nodes(source, observed=observed)
independent = set(asia.nodes()) - {source}
independent - set(observed) - set(
                        asia.active_trail_nodes(source, observed=observed)[source])
