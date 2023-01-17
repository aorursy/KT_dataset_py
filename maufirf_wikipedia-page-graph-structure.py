# AUTOLAB_IGNORE_START
import networkx as nx
# AUTOLAB_IGNORE_STOP
nxwiki_edges = nx.read_edgelist('../input/wikipedia_small.graph')
nxwiki_nodes = [line.replace('\n','') for line in open('../input/wikipedia_small.nodes','r')]
#for x in nxwiki_nodes: print(x)
nxwiki_sp = nx.shortest_path_length(nxwiki_edges,str(nxwiki_nodes.index('Carnegie_Mellon_University')))
o=0
for entry in nxwiki_sp:
    print(nxwiki_nodes[int(entry)],':',nxwiki_sp[entry])
    o+=1
    if o>=10: break
nxwiki_abpath = nx.shortest_path(nxwiki_edges,str(nxwiki_nodes.index('Carnegie_Mellon_University')),str(nxwiki_nodes.index('List_of_Salticidae_species')))
for entry in nxwiki_abpath:
    print(nxwiki_nodes[int(entry)])
import numpy as np
import scipy.sparse as sp
import heapdict

class Graph:
    def __init__(self):
        """ Initialize with an empty edge dictionary. """
        self.edges = {}
        #self-add
        self.nodes = []
    
    def add_edges(self, edges_list):
        """ Add a list of edges to the network. Use 1.0 to indiciate the presence of an edge. 
        
        Args:
            edges_list: list of (a,b) tuples, where a->b is an edge to add
        """
        for edge in edges_list:
            if edge[0] not in self.edges:
                self.edges[edge[0]]={edge[1]:1}
            else:
                self.edges[edge[0]][edge[1]]=1
            if edge[1] not in self.edges:
                self.edges[edge[1]]={}
            if edge[0] not in self.nodes:
                self.nodes.append(edge[0])
            if edge[1] not in self.nodes:
                self.nodes.append(edge[1])
                
        pass
        
    def shortest_path(self, source):
        """ Compute the single-source shorting path.
        
        This function uses Djikstra's algorithm to compute the distance from 
        source to all other nodes in the network.
        
        Args:
            source: node index for the source
            
        Returns: tuple: dist, path
            dist: dictionary of node:distance values for each node in the graph, 
                  where distance denotes the shortest path distance from source
            path: dictionary of node:prev_node values, where prev_node indicates
                  the previous node on the path from source to node
        """
        mark=[x for x in self.nodes]
        queue=[source]
        dist={}
        prev={}
        for node in self.nodes:
            dist[node]=np.inf
            prev[node]=None
        dist[source]=0
        for parent in queue:
            for child in self.edges[parent]:
                if child in mark:
                    distance = self.edges[parent][child]+dist[parent]
                    if distance <= dist[child]:
                        dist[child]=self.edges[parent][child]+dist[parent]
                        prev[child]=parent
                        if child not in queue:
                            queue.append(child)
        return dist,prev
        pass
    
        
    def adjacency_matrix(self):
        """ Compute an adjacency matrix form of the graph.  
        
        Returns: tuple (A, nodes)
            A: a sparse matrix in COO form that represents the adjaency matrix
               for the graph (i.e., A[j,i] = 1 iff there is an edge i->j)
               NOTE: be sure you have this ordering correct!
            nodes: a list of nodes indicating the node key corresponding to each
                   index of the A matrix
        """
        data = []
        i = []
        j = []
        nodes = [x for x in self.edges]
        for source in nodes:
            for target in self.edges[source]:
                data.append(1)
                j.append(nodes.index(source))
                i.append(nodes.index(target))
        return (sp.coo_matrix((data, (i,j)), shape=(len(nodes),len(nodes))),nodes)
        #adjacency_matrix = np.zeros((len(self.nodes),len(self.nodes)))
        #for source_idx in range(len(self.nodes)):
        #    for destination in self.edges[self.nodes[source_idx]]:
        #        adjacency_matrix[self.nodes.index(destination)][source_idx]=1
        #return sp.coo_matrix(adjacency_matrix),[x for x in self.nodes]
        pass
    
    def pagerank(self, d=0.85, iters=100):
        """ Compute the PageRank score for each node in the network.
        
        Compute PageRank scores using the power method.
        
        Args:
            d: 1 - random restart factor
            iters: maximum number of iterations of power method
            
        Returns: dict ranks
            ranks: a dictionary of node:importance score, for each node in the
                   network (larger score means higher rank)
        
        """
        adjacency_matrix_t = np.zeros((len(self.nodes),len(self.nodes)))
        ones = np.ones((len(self.nodes),len(self.nodes)),dtype=np.float32)
        for source_idx in range(len(self.nodes)):
            for destination in self.edges[self.nodes[source_idx]]:
                adjacency_matrix_t[source_idx][self.nodes.index(destination)]=1
        for col in adjacency_matrix_t:
            if 1 not in col: col.fill(1)
            col_proba = 1/sum(col)
            col[col==1]=col_proba
        p_cap = ((d)*adjacency_matrix_t.transpose())+(((1-d)/len(self.nodes))*ones*ones.transpose())
        x = (1/(len(self.nodes)))*np.ones((len(self.nodes),1),dtype=np.float32)
        for i in range(iters):
            x = p_cap@x
        return {self.nodes[i]:x[i][0] for i in range(len(self.nodes))}
        pass
    
# HANDOUT_END   

# AUTOLAB_IGNORE_START
G = Graph()
G.add_edges([("A","B"), ("B","C"), ("A","D"), ("D", "E"), ("E", "B")])
#G.add_edges([("A","B"), ("B","C"), ("C","A"), ("C", "D")])
print (G.edges)
# AUTOLAB_IGNORE_STOP
from time import time

t_now = int(time())
edges_file = open('../input/wikipedia_small.graph', 'r')
entries = [line.replace('\n','').split() for line in edges_file]
print("load to array done in",(int(time())-t_now),"seconds")
t_now = int(time())
gwiki=Graph()
gwiki.add_edges(entries)
print("initializing array done in",(int(time())-t_now),"seconds")
o=0
for source in gwiki.edges:
    print(source,':',gwiki.edges[source])
    o+=1
    if o >= 10: break
# AUTOLAB_IGNORE_START
G_sp = G.shortest_path("A")
print(G_sp)
# AUTOLAB_IGNORE_STOP
import operator
for entry in sorted(G_sp[0].items(), key=operator.itemgetter(1)):
    print(entry[0],':',entry[1])
gwiki_nodes = [line.replace('\n','') for line in open('../input/wikipedia_small.nodes','r')]
gwiki_sp = gwiki.shortest_path(str(gwiki_nodes.index('Carnegie_Mellon_University')))
o=0
for entry in sorted(gwiki_sp[0].items(), key=operator.itemgetter(1)):
    print(gwiki_nodes[int(entry[0])],':',entry[1])
    o+=1
    if o >= 10: break
# AUTOLAB_IGNORE_START
A, Anlist = G.adjacency_matrix()
print (type(A))
print (A.todense())
print (Anlist)
# AUTOLAB_IGNORE_STOP
B, Bnlist = Gr.adjacency_matrix()
print (type(B))
print (B.todense())
print (Bnlist[:10])
# AUTOLAB_IGNORE_START
G_pgr = G.pagerank()
print(G_pgr)
# AUTOLAB_IGNORE_STOP
for entry in sorted(G_pgr.items(), reverse=true, key=operator.itemgetter(1)):
    print(entry[0],entry[1])
gwiki_pgr = gwiki.pagerank()
print(len(gwiki_pgr))
o=0
for entry in sorted(gwiki_pgr.items(), reverse=true, key=operator.itemgetter(1)):
    o+=1
    print(gwiki_nodes[int(entry[0])],entry[1])
    if o>=10: break
