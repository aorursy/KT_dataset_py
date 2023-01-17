# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import networkx as nx

from networkx.algorithms import community

import numpy as np

import matplotlib.pyplot as plt 

import random

plt.rcParams.update({'figure.max_open_warning': 50})



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



def plot_degree_dist(G):

    """ Generates a plot with the degrees of distribution of the connected components. 

    To facilitate the representation it was decided to also use the loglog contained in numpy

    

    --------------------------------------------------

    

    Input: G---> Graphs

               A networkx graph



    

    Output: a list of values of the degree distribution

        

        """

    degrees = G.degree()

    degrees = dict(degrees)

    values = sorted(set(degrees.values()))

    histo = [list(degrees.values()).count(x) for x in values]

    P_k = [x / G.order() for x in histo]

    

    plt.figure()

    plt.plot(values, P_k, "ro-")

    plt.xlabel("k")

    plt.ylabel("p(k)")

    plt.title("Degree Distribution")

    plt.show()

    

    plt.figure()

    plt.grid(False)

    plt.loglog(values, P_k, "bo-")

    plt.xlabel("log k")

    plt.ylabel("log p(k)")

    plt.title("log Degree Distribution")

    plt.show()

    

    """"Plot of the histogram degree distribution"""

    

    

    plt.figure()

    degrees = [G.degree(n) for n in G.nodes()]

    counts = dict()

    for i in degrees:

        counts[i] = counts.get(i, 0) + 1

    axes = plt.gca()

    axes.set_xlim([0,100])

    axes.set_ylim([0,1000])

    plt.grid(False)

    plt.bar(list(counts.keys()), counts.values(), color='r')

    plt.title("Degree Histogram")

    plt.ylabel("Count")

    plt.xlabel("Degree")

    plt.show()

    

def plot_degree_In(G):

 

    """ Generates a plot with the  IN/OUT degrees of distribution of the connected components. 

    To facilitate the representation it was decided to also use the loglog contained in numpy

    

        --------------------------------------------------

     Parameters

    

    

    Input: G---> Graphs

    

           A networkx graph



    

    Output: a list of values of the degree distribution

    

    """

    N = G.order()

    in_degrees = G.in_degree()  #built-in function to estimate in-degree distribution

    in_degrees = dict(in_degrees)

    in_values= sorted(set(in_degrees.values()))

    in_hist = [list(in_degrees.values()).count(x) for x in in_values]

    in_P_k = [x / N for x in in_hist]

    out_degrees = G.out_degree()   #built-in function to estimate out-degree distribution

    out_degrees = dict(out_degrees)

    out_values = sorted(set(out_degrees.values()))

    out_hist = [list(out_degrees.values()).count(x) for x in out_values]

    out_P_k = [x / N for x in out_hist]

    

    plt.figure()

    plt.grid(False)

    plt.plot(in_values ,in_P_k, "r.")

    plt.plot(out_values,out_P_k, "b.")

    plt.legend(['In-degree','Out-degree'])

    plt.xlabel("k")

    plt.ylabel("p(k)")

    plt.title("Degree Distribution")

    plt.show()

    

    plt.figure()

    plt.grid(False)

    plt.loglog(in_values ,in_P_k, "r.")

    plt.loglog(out_values,out_P_k, "b.")

    plt.legend(['In-degree','Out-degree'])

    plt.xlabel("log k")

    plt.ylabel("log p(k)")

    plt.title("log log Degree Distribution")

    plt.show()

       

def plot_clustering_coefficient(G):

        """ Generates a plot with the  clustering coefficientof.It is a measure of the degree to which

        nodes in a graph tend to cluster together. 

    To facilitate the representation it was decided to also use the loglog contained in numpy

        --------------------------------------------------

     Parameters

    Input: G---> Graphs

    

    Output: a list of values of the degree distribution

    """

        clust_coefficients = nx.clustering(G)  #built-in function to estimate clustering coeff  

        clust_coefficients = dict(clust_coefficients)

        values1= sorted(set(clust_coefficients.values()))

        histo1 = [list(clust_coefficients.values()).count(x) for x in values1]

        

        plt.figure()

        plt.grid(False)

        plt.plot(values1,histo1, "r.")

        plt.xlabel("k")

        plt.ylabel("C (Clustering Coeff)")

        plt.title("Clustering Coefficients")

        plt.show()

        

        

        plt.figure()

        plt.grid(False)

        plt.loglog(values1,histo1, "r.")

        plt.xlabel("log degree k")

        plt.ylabel("c (clustering coeff)")

        plt.title("log log Clustering Coefficients")

        plt.show()

        

        plt.figure()

        degrees1 = [nx.clustering(G,n) for n in G.nodes()]

        plt.hist(degrees1)

        plt.xlabel("log degree k")

        plt.ylabel("C (Clustering Coeff) hist")

        plt.title("Clustering Coefficients")

        plt.show()

    

def plot_shortest_path_length(G):

    

    """Plot and Compute shortest paths in the graph.



    Parameters

    ----------

    G : NetworkX graph





    Returns

    -------

    path: list or dictionary

        All returned paths include both the source and target in the path.



        If the source and target are both specified, return a single list

        of nodes in a shortest path from the source to the target.



        If only the source is specified, return a dictionary keyed by

        targets with a list of nodes in a shortest path from the source

        to one of the targets.



        If only the target is specified, return a dictionary keyed by

        sources with a list of nodes in a shortest path from one of the

        sources to the target.



        If neither the source nor target are specified return a dictionary

        of dictionaries with path[source][target]=[list of nodes in path].

"""

    

    dist = {}   #inizialize distance dictionary

    len_pathlengths = 0    #inizialize length path

    sum_pathlengths = 0    #inizialize sum of length path

    for n in G.nodes():

        pathlenghts = []

        spl = nx.single_source_shortest_path_length(G,n)

        

        """Compute the shortest path lengths from source to all reachable nodes.

        Parameters:

            •G (NetworkX graph) – 

            •source (node) – Starting node for path

            •cutoff (integer, optional) – Depth to stop the search. Only paths of length <= cutoff are returned.

        Returns:

            lengths – Dictionary of shortest path lengths keyed by target.

            

        Return type: dictionary

 



"""

        for p in spl:

            pathlenghts.append(spl[p])

            len_pathlengths += 1

            

        for p in pathlenghts:

            if p in dist:

                dist[p] +=1

            else:

                dist[p] = 1        

        sum_pathlengths += sum(pathlenghts)

            

    print('')

    print("Average shortest path length %s" % (sum_pathlengths/ len_pathlengths))  

    print("diameter: %d" % nx.diameter(max(nx.connected_component_subgraphs(G), key=len)))  

    connected_components = [len(c_components) for c_components in sorted(nx.connected_components(G), key = len, reverse  = True)]

    print("connected components: %s" % connected_components)    

    

    print('')

    print("length #paths")

          

    for d in sorted(list(dist.keys())[1:]):  

        print('%s %d ' % (d, dist[d]))

    

    length_path = list (dist.keys())[1:] 

    no_paths = list (dist.values()) [1:]

    

    plt.figure()

    plt.grid(False)

    plt.plot(length_path, no_paths, 'ro-')

    plt.xlabel('distance')

    plt.ylabel('number of paths')

    plt.title('Length of shortest paths' )

    plt.show()

#%%   ---------------------------------------SMALL WORLD UTILITIES-----------------------------------------



    

def adjacent_edges(nodes, halfk):    



    n = len(nodes)

    for i, u in enumerate(nodes):

        for j in range(i+1, i+halfk+1):

            v = nodes[j % n]

            yield u,v





       

def make_ring_lattice(n,k):

    """Generate a view of lattice graph"

            Parameters:

            •G (NetworkX graph) – 

            

            k (node) – number of adjacent nodes

        Returns:

            graphs – a graph lattice view

            

        Return type: networkx graph

    """

    G = nx.Graph()

    nodes = range(n)

    G.add_nodes_from(nodes)

    G.add_edges_from(adjacent_edges(nodes, k//2))

    return G



def flip(p):

    return np.random.random() < p 



def rewire(G,p):

    nodes = set(G)

    for u, v in G.edges():

        if flip(p): 

            choices = nodes - {u} - set(G[u])

            new_v = np.random.choice(list(choices))

            G.remove_edge(u, v)

            G.add_edge(u, new_v)

            

def small_world(n,k,p):

    sw = make_ring_lattice(n,k)

    rewire(sw,p)

    return sw
def k_core(G,k,t):

    

    

        """Return the core number for each vertex.



    A k-core is a maximal subgraph that contains nodes of degree k or more.



    The core number of a node is the largest value k of a k-core containing

    that node.



    Parameters

    ----------

    G : NetworkX graph

       A graph or directed graph



    Returns

    -------

    core_number : dictionary

       A dictionary keyed by node to the core number"""

       

        H=G.copy() 

        i=1

        while (i>0):

            i=0

            for node in list(H.nodes()):

                if H.degree(node)<k:

                    H.remove_node(node)

                    i+=1

        if (H.order()!=0):

            plt.figure()

            plt.title(str(k) +'-core decomposition of' + t) 

            nx.draw(H,with_labels=True)

        return H



def full_k_core_decomposition(G,t):

    empty = False

    k=1

    while (empty==False):

        H = k_core(G,k,t)

        k+=1

        if (H.order()==0):

            empty = True

            
graphs = nx.read_edgelist('../input/com-youtube.ungraph.txt',create_using=nx.Graph(), nodetype=int) 



subset = 1000

edges = graphs.edges()

edges = list(edges)[:int(subset)]   

edges = [list(elem) for elem in edges] 





#%% formatting necessary to allow performing nx.parse_edglist



_newlist = []

_list = []



for subsets in edges:

    for element in subsets:

        _list.append(element)



_temp= int(len(_list)*0.5)



for i in range (_temp):

    _newlist.append(str(_list[2*i]) + " " + str(_list[2*i +1]) ) 

print(_newlist)

graphs = nx.parse_edgelist(_newlist, nodetype = int) 



#%%

"""1 Original Graphs Measures"""

N=graphs.order()  

E = graphs.number_of_edges()  

Av_deg_undirected = float(2*E)/N  



print ("\n ORIGINAL GRAPH: ")

print("The number of nodes is:", N)

print("The number of edges is:", E)

print("The average degree (undirected graph) is:", Av_deg_undirected)



plot_degree_dist(graphs) 

plot_clustering_coefficient(graphs)

plot_shortest_path_length(graphs)

print ('The average clustering coefficient is: ' + str(nx.average_clustering(graphs)))
communities_gen = community.girvan_newman(graphs)

top_level_communities = next(communities_gen)

next_level_communities = next(communities_gen)

a=sorted(map(sorted,next_level_communities))
#%% 

"""GENERATION OF THE SBM GRAPH"""



sizes = []

probs = []









for com in a:

  sizes.append(len(com))

  



num11 = sizes[0] * (sizes[0]-1)*0.5

num12 = sizes[0] * sizes[1]

num13 = sizes[0] * sizes[2]

num22 = sizes[1] * (sizes[1]-1)*0.5

num23 = sizes[1] * sizes[2]

num33 = sizes[2] * (sizes[2]-1)*0.5



num_edges11,num_edges22,num_edges33,num_edges12,num_edges13,num_edges23 = [0,0,0,0,0,0]

for g in edges:

    g[0] = int(g[0])

    g[1] = int(g[1]) 

for h in edges:

  if (h[0] in a[0] and h[1] in a[0]):

    num_edges11+=1

  elif (h[0] in a[1] and h[1] in a[1]):

    num_edges22+=1

  elif (h[0] in a[2] and h[1] in a[2]):

    num_edges33+=1    

  elif ((h[0] in a[0] and h[1] in a[1]) or (h[0] in a[1] and h[1] in a[0])):

    num_edges12+=1    

  elif ((h[0] in a[0] and h[1] in a[2]) or (h[0] in a[2] and h[1] in a[0])):

    num_edges13+=1

  else:

      (h[0] in a[1] and h[1] in a[2]) or (h[0] in a[2] and h[1] in a[1])

      num_edges23+=1    

p11 = float (num_edges11/num11)

p12 = float (num_edges12/num12)

p13 = float (num_edges13/num13)

p22 = float (num_edges22/num22)

p23 = float (num_edges23/num23)

p33 = float (num_edges33/num33)



probs1 = [p11,p12,p13],[p12,p22,p23],[p13,p23,p33]



print(probs1)

#%% It is now possible to generate the SBM and calculate some statistic measures on it



SBM = nx.stochastic_block_model(sizes, probs1, seed=0)



"""2a STATISTICS ABOUT MEASURES - SBM"""

SBM_nodes=SBM.order()

SBM_Edges = SBM.number_of_edges()

Av_deg_SBM = float(2*SBM_Edges)/SBM_nodes



print ("\n SBM GRAPH: ")

print("The number of nodes is:", SBM_nodes)

print("The number of edges is:", SBM_Edges)

print("The average degree (undirected graph) is:", Av_deg_SBM)



plot_degree_dist(SBM) 

plot_clustering_coefficient(SBM)

plot_shortest_path_length(SBM)

print ('The average clustering coefficient is: ' + str(nx.average_clustering(SBM)))
#%% Generation of Erdos-Renyi random graph



"""2b STATISTICS ABOUT MEASURES - ERDOS-RENYI"""



Proba = E/(N*(N-1)/2)

Erdos_renyi = nx.erdos_renyi_graph (N, Proba)

Nodes_erdos=Erdos_renyi.order()

Edges_erdos = Erdos_renyi.number_of_edges()

Av_deg_und_erdos = float(2*Edges_erdos)/Nodes_erdos  



print ("\n ERDOS-RENYI GRAPH: ")

print("The number of nodes is:", Nodes_erdos)

print("The number of edges is:", Edges_erdos)

print("The average degree (undirected graph) is:", Av_deg_und_erdos)



plot_degree_dist(Erdos_renyi) 

plot_clustering_coefficient(Erdos_renyi)

plot_shortest_path_length(Erdos_renyi)

print ('The average clustering coefficient is: ' + str(nx.average_clustering(Erdos_renyi)))

"""2b STATISTICS ABOUT MEASURES - SMALL WORLD"""



Small_World = small_world(N,4,0.2)

nx.draw_circular(Small_World)



Nodes_SW=Small_World.order()

Edge_SW = Small_World.number_of_edges()

Av_deg_SW = float(2*Edge_SW)/Nodes_SW  

print ("\n SMALL WORLD GRAPH: ")

print("The number of nodes is:", Nodes_SW)

print("The number of edges is:", Edge_SW)

print("The average degree (undirected graph) is:", Av_deg_SW)



plot_degree_dist(Small_World) 

plot_clustering_coefficient(Small_World)

plot_shortest_path_length(Small_World)

print ('The average clustering coefficient is: ' + str(nx.average_clustering(Small_World)))
"""4 DIRECTED VERSION WITHOUT 25% OF THE LINKS"""



k = int(E*0.25)

DG = graphs.copy()

DG = DG.to_directed()

edges_d = DG.edges()

list_edges_d = list(edges_d)

random.shuffle(list_edges_d)



for edgee in list_edges_d:

    if (DG.degree(edgee[0]) != 1 and DG.degree(edgee[1]) != 1):

        DG.remove_edge(edgee[0],edgee[1])

        k-=1

    if (k==0):

        break



N4=DG.order()

E4 = DG.number_of_edges()

Av_deg_d = float(E)/N 



print ("\n DIRECTED VERSION WITHOUT 25% OF THE LINKS: ")

print("The number of nodes is:", N4)

print("The number of edges is:", E4)

print("The average degree (directed graph) is:", Av_deg_d)



plot_degree_dist(DG)

plot_degree_In(DG)

plot_clustering_coefficient(DG)

DG = DG.to_undirected()

plot_shortest_path_length(DG)

print ('The average clustering coefficient is: ' + str(nx.average_clustering(DG)))

page_rank = nx.pagerank(graphs)

dict_degree_centrality = nx.degree_centrality(graphs)

dict_closeness_centrality = nx.closeness_centrality(graphs)

#dict_eigenvector_centrality = nx.eigenvector_centrality(graphs)   #sometimes doesn't run

dict_harmonic_centrality = nx.harmonic_centrality(graphs)

dict_betweeness=nx.betweenness_centrality(graphs)
"""6 K-CORE DECOMPOSITION"""   

_G_CORE = nx.k_core(graphs, 2)

pos = nx.spring_layout(graphs)

plt.figure()

plt.title(' networkx 2-core decomposition of Original graph')

nx.draw_networkx(_G_CORE , pos = pos, node_size = 1, edge_color = "blue", alpha = 0.5, with_labels = True)

Original_Graph = full_k_core_decomposition(graphs, ' Original graph')

SBM_graph = full_k_core_decomposition(SBM, ' Stochastic Block Model graph')

Erdos_Renyi_graph = full_k_core_decomposition(Erdos_renyi, ' Erdos Renyi graph')

Smal_Word_graph = full_k_core_decomposition(Small_World, ' Small World graph')

Degraded_Graphs = full_k_core_decomposition(DG, ' degraded graph')

  