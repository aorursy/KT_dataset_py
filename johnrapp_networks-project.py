# Read data



import json

import networkx as nx

from networkx.algorithms import approximation as nxa

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

import random

import copy

import matplotlib.ticker as ticker



with open('../input/networksprojectdataset/lines.json') as f:

    lines = json.load(f)
# distances between nodes that will be used as weights

# could be cleaner if these were somehow formatted into json file -Jack



line_1_distances = [.3, .1, .2, .2, .1, .2, .2, .1, .2, .2, .1, 

                    .1, .1, .2, .1, .2, .1, .2, .2, .3, .2, .2, 

                    .3, .2, .1, .3, .1, .1, .3, .2, .1, .3, .1, 

                    .2, .3, .2, .1, .2, .3]



line_2_distances = [.8, .4, .1, .3, .1, .2, .3, .1, .2, .1, .2, 

                    .2, .1, .3, .1, .1, .3, .2, .1, .1, .2, .2, 

                    .3]



line_3_distances = [1, .2, .2, .1, .1, .2, .2, .3, .5, .3, .1, 

                    .2, .3, .1, .1, .2, .4, .6, .1, .4, .1, .2, 

                    .3, .2, .1, .1, .1, .1, .1, .1, .2, .2, .3,

                    .2, .3, .2, .2, .1, .1, .3, .2, .1, .4, .2]



line_4_distances = [.4, .5, .2, .2, .9, .2, .2, .2, .2, .5, .6, 

                    .3, .1, .1, .1, .1, .8, .6, .2, .2, .3, .3, 

                    .1, .3, .4, .1, .2, .3]



line_5_distances = [.6, .3, .5, .2, .3, .2, .4, .3, .1, .2, .1, 

                    .1, .1, .2, .2, .2]



line_6_distances = [.2, .2, .5, .3, .1, .3, .3, .3, .3, .4, .2, 

                    .2, .2, .2, .2, 1, .2, .1, .6, .3, .3, .1, .1,

                    .2, .2, .3, .2, .2, .1, .3, .4, .2, .4, .3, 

                    .2, .2, .2, .1, .3, .2, .4, .2, .2, .5, .3, 

                    .2, .2, .2]



line_7_distances = [.2, .2, .1, .2, .2, .1, .3, .3, .3, .2, .2, .2, 

                    .2, .1, .3, .3, .2, .2, .3, .1, .1, .3, .2,

                    .3, .1, .3, .1, .2, .6]



line_distances = {'LINE 1': line_1_distances,

                  'LINE 2': line_2_distances,

                  'LINE 3': line_3_distances,

                  'LINE 4': line_4_distances,

                  'LINE 5': line_5_distances,

                  'LINE 6': line_6_distances,

                  'LINE 7': line_7_distances}
# Convert to network

G = nx.Graph()   # main graph with all nodes combined

G_nodes_list = []   # list of all nodes

extra_list = [];

# Add nodes

for line_name, nodes in lines.items():   

    for node in nodes:

        if(node not in G_nodes_list):  # adding nodes to nodes_list

            G_nodes_list.append(node)



        G.add_node(node)  # adding nodes to G

        

        # adding list of lines that nodes are in as node attribute

        if('line' not in G.nodes[node].keys()):

            G.nodes[node]['line'] = [line_name]

        else:

            G.nodes[node]['line'].append(line_name)

            

    # adding edges with weights and line number    

    for i in range(1, len(nodes)):

        G.add_edge(nodes[i - 1], nodes[i], weight=line_distances[line_name][i-1], 

                   line=line_name)

        

print("Nr nodes: ", len(G.nodes))

print("Nr edges: ", len(G.edges))
# Draw network



random.seed(2020)



pos = nx.spring_layout(G)



nx.draw(G, pos)
# finding number of different buses needed to go from one station to another



# path creates nested dict with shortest paths for any node to any other node

path = dict(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path(G))



bus_lines = []

route = path['FAIRFAX A / WASHINGTON B']['BRADDOCK DR / INGLEWOOD B']



for i in range(1, len(route)):

    line = G[route[i-1]][route[i]]['line']  # bus line used to go from one station to next

    

    if(line not in bus_lines):  # adding bus line to list if it is used in shortest path and not already in the list

        bus_lines.append(line)

            



print('Starting node: ' + route[0]) 

print('Ending node: ' + route[-1])

print('Different lines: ' + str(bus_lines))

print('Num of buses: ' + str(len(bus_lines)))
# finding avg. number of different buses needed to go from any station to any other



def avg_buses(G, nodes_list):

    """Finds avg num of buses for all stations

    

       Args:

           G: Graph

    """

    path = dict(nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path(G))

    total_routes = 0

    total_buses = 0;



    # iterating through all possible starting and ending nodes (doesn't include path from node to self)

    for i in range(0, len(nodes_list)-1):

        for j in range(i+1, len(nodes_list)):



            total_routes += 1

            bus_lines = []

            route = path[nodes_list[i]][nodes_list[j]] # list of nodes between starting and ending node        



            for n in range(1, len(route)):                            

                line = G[route[n-1]][route[n]]['line']   # bus line used to go from one station to next



                if(line not in bus_lines):   # adding bus line to list if it is used in shortest path and not already in the list

                    bus_lines.append(line)



            total_buses += len(bus_lines)





    #print('Num of routes: ' + str(total_routes))

    #print('Num of buses: ' + str(total_buses))

    print('Avg. num of buses: ' + str(total_buses/total_routes))

    #print(" ")
# finding avg. number of stops needed to go from any station to any other



def avg_stops(G, nodes_list):

    """Finds avg num of stops for all stations

    

       Args:

           G: Graph

           nodes_list: nodes in G

    """

    path = dict(nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path(G))

    total_routes = 0

    total_stops = 0;



    # iterating through all possible starting and ending nodes (doesn't include path from node to self)

    for i in range(0, len(nodes_list)-1):

        for j in range(i+1, len(nodes_list)):

            total_routes += 1

            route = path[nodes_list[i]][nodes_list[j]]   # list of nodes between starting and ending node             

            total_stops += len(route)-1  # num of stops needed (subtracts 1 because route includes starting node)





    #print('Num of routes: ' + str(total_routes))

    #print('Num of stops: ' + str(total_stops))

    print('Avg. num of stops: ' + str(total_stops/total_routes))

    #print(" ")
# finding avg. number of miles needed to go from any station to any other



def avg_distance(G, nodes_list):

    """Finds avg distance (miles) between all stations

    

       Args:

           G: Graph

           nodes_list: nodes in G

    """

    path = dict(nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path_length(G))

    total_routes = 0

    total_miles = 0;



    # iterating through all possible starting and ending nodes (doesn't include path from node to self)

    for i in range(0, len(nodes_list)-1):

        for j in range(i+1, len(nodes_list)):

            total_routes += 1

            route = path[nodes_list[i]][nodes_list[j]]   # distance between one node and another 

            total_miles += route  # total_miles of all shortest path lengths





    #print('Num of routes: ' + str(total_routes))

    #print('Num of Miles: ' + str(total_miles))

    print('Avg. num of miles: ' + str(total_miles/total_routes))

    #print(" ")
# finding avg. number of miles needed to go from one station to all others



def avg_distance_single(G, node, nodes_list):

    """Finds avg distance (miles) between all stations

    

       Args:

           G: Graph

           node: node to find average distance

           nodes_list: list of all nodes in graph

    """

    path = dict(nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path_length(G))

    total_routes = 0

    total_miles = 0;



    # iterating through all possible starting and ending nodes (doesn't include path from node to self)

    for j in range(0, len(nodes_list)):

        if(nodes_list[j] != node):

            total_routes += 1

            route = path[node][nodes_list[j]]   # distance between one node and another 

            total_miles += route  # total_miles of all shortest path lengths





    #print('Num of routes: ' + str(total_routes))

    #print('Num of Miles: ' + str(total_miles))

    print('Avg. num of miles: ' + str(total_miles/total_routes))

    #print(" ")

    

def remove_close_nodes(G, nodes_list):

    """Will remove nodes from graph if they only belong to 1 line and have short edges

    

       Args:

           G: Graph

       Returns:

           L: Graph with nodes removed

    """

    L = copy.deepcopy(G)  # says deep copy but really a shallow copy. G doesn't change when L changes

    n = copy.copy(nodes_list)   # shallow copy of list of nodes. Will be returned after it is updated when nodes removed



    for line_name, stops in lines.items(): 

        i = 1;

        s = copy.copy(stops)  # shallow copy



        while(i<len(s)-1):



            a = L.edges[s[i-1],s[i]]['weight']   # distance between nodes i-1 and i

            b = L.edges[s[i],s[i+1]]['weight']



            if(a+b<.35 and len(L.nodes[s[i]]['line'])<1.5):  # true if nodes adjacent edges have distance <= .3 and node only on one line

                L.add_edge(s[i-1], s[i+1],  weight=(a+b), line=line_name)

                L.remove_node(s[i])

                n.remove(s[i])

                s.remove(s[i])



            else:

                i+=1  

                

    return L, n
def add_line(G, bus_line, line_distances, nodes_list):

    """Will create a new network with new bus line given data for new line

    

       Args:

           G: Graph

           bus_line: dict with key as line name, value as nodes

           line_distances: distances between nodes

           nodes_list: list of nodes, new list with updated list of nodes will be returned

       Returns:

           L: new Graph with new bus line included

           n: nodes list associated with L

    """

    L = copy.deepcopy(G)  # says deep copy but really a shallow copy. G doesn't change when L changes

    n = copy.copy(nodes_list)   # shallow copy of list of nodes. Will be returned after it is updated when nodes removed

    

    

    

    for line_name, nodes in bus_line.items():   

        for node in nodes:

            if(node not in n):  # adding nodes to nodes_list

                n.append(node)



            L.add_node(node)  # adding nodes to G



            # adding list of lines that nodes are in as node attribute

            if('line' not in L.nodes[node].keys()):

                L.nodes[node]['line'] = [line_name]

            else:

                L.nodes[node]['line'].append(line_name)



        # adding edges with weights and line number    

        for i in range(1, len(nodes)):

            L.add_edge(nodes[i - 1], nodes[i], weight=line_distances[i-1], line=line_name) 

                

    return L, n
def remove_line(bus_line):

    """Will create a new network by leaving out a bus line

    

       Args:

           G: Graph

           bus_line: dict with key as line name, value as nodes

           nodes_list: list of nodes, new list with updated list of nodes will be returned

       Returns:

           L: new Graph with new bus line included

           n: nodes list associated with L

    """



    G = nx.Graph()   # main graph with all nodes combined

    nodes_list = []   # list of all nodes

    extra_list = [];

    # Add nodes

    for line_name, nodes in lines.items(): 

        if(line_name != bus_line):

            for node in nodes:

                if(node not in nodes_list):  # adding nodes to nodes_list

                    nodes_list.append(node)



                G.add_node(node)  # adding nodes to G



                # adding list of lines that nodes are in as node attribute

                if('line' not in G.nodes[node].keys()):

                    G.nodes[node]['line'] = [line_name]

                else:

                    G.nodes[node]['line'].append(line_name)



            # adding edges with weights and line number    

            for i in range(1, len(nodes)):

                G.add_edge(nodes[i - 1], nodes[i], weight=line_distances[line_name][i-1], 

                           line=line_name)

    return G, nodes_list



    

    
def print_metrics(line_removed):

    

    # graph without line 1

    G, nodes_list = remove_line(line_removed)



    print("Metrics for graph without " + line_removed)

    print("Graph Connected: " + str(nx.is_connected(G)))

    print("Num 2-components " + str(len(nxa.k_components(G)[2])))

    print("Length 2-component " + str(len(nxa.k_components(G)[2][0])))

    avg_stops(G, nodes_list)

    avg_buses(G, nodes_list)

    avg_distance(G, nodes_list)

    print("")
print_metrics("LINE 1")

print_metrics("LINE 2")

print_metrics("LINE 3")

print_metrics("LINE 4")

print_metrics("LINE 5")

print_metrics("LINE 6")

print_metrics("LINE 7")

L, L_nodes_list = remove_close_nodes(G, G_nodes_list)   # creating graph and list of its nodes by removing close nodes



# comparing new graph to original, numbers look better

avg_stops(G, G_nodes_list)

avg_buses(G, G_nodes_list)

avg_distance(G, G_nodes_list)

avg_stops(L, L_nodes_list)

avg_buses(L, L_nodes_list)

avg_distance(L, L_nodes_list)

len(L_nodes_list)

additional_line = {"LINE 8": ['EXPO LIGHT RAIL STATION / WASHINGTON B', 'WASHINGTON B/INCE B', "CULVER B / MAIN ST (Menchie's)", 

                              'MOTOR A / WASHINGTON B', 'OVERLAND A / WASHINGTON B', 'SEPULVEDA B / WASHINGTON B', 

                              'INGLEWOOD B / WASHINGTON B', 'WASHINGTON B / WASHINGTON PL', 'BEETHOVEN ST / WASHINGTON B', 

                              'GLENCOE A / WASHINGTON B', 'JEFFERSON B / MESMER A', 'CULVER B / INGLEWOOD B', 

                              'CULVER CITY TRANSIT CENTER', 'JEFFERSON B / OVERLAND A', 'BRADDOCK DR / OVERLAND A', 

                              'CULVER B / OVERLAND A', 'SAWTELLE B / SEPULVEDA B', 'CULVER B / DUQUESNE A', 

                              'CULVER B / MADISON A', 'CULVER B / SEPULVEDA B']}



a_l = {"LINE 1345_CONNECTER": ['DUQUESNE A / JEFFERSON B', 'BRADDOCK DR / MADISON A', 

                   'OVERLAND A / WASHINGTON B', ]}

a_l_distances =[.6, .7]

for line, names in  additional_line.items():

    for na in names:

        

        print(str(na) + str(G.nodes[na]))



M, M_nodes_list = add_line(G, a_l, a_l_distances, G_nodes_list)
print("Nr nodes: ", len(M.nodes))

print("Nr edges: ", len(M.edges))

avg_stops(M, M_nodes_list)

avg_buses(M, M_nodes_list)

avg_distance(M, M_nodes_list)

k_comps = nxa.k_components(M)

print(len(k_comps[2]))
# Centrality histograms



unnormalized_dc = {name: round(c*(len(G_nodes_list)-1)) for name, c in nx.degree_centrality(G).items()} # unnormailzed degree centrality



   



plt.figure(0)

_ = plt.hist(list(unnormalized_dc.values()), bins=[0,1,2,3,4,5,6,7], rwidth=.5, align='left')                   

_ = plt.title('Degree centrality distribution')



plt.figure(1)

_ = plt.hist(list(nx.closeness_centrality(G).values()))

_ = plt.title('Closeness centrality distribution')



plt.figure(2)

_ = plt.hist(list(nx.betweenness_centrality(G).values()))

_ = plt.title('Betweenness centrality distribution')
# Highest centrality nodes



def highest_vals(d, qq):

    q = np.quantile(list(d.values()), qq)

    return list(map(lambda item: item[0], filter(lambda item: item[1] > q, d.items())))

    

print("Highest degree centrality", highest_vals(nx.degree_centrality(G), 0.8))

print('')

print("Highest closeness centrality (stops)", highest_vals(nx.closeness_centrality(G), 0.95))

print('')

print("Highest closeness centrality (distance)", highest_vals(nx.closeness_centrality(G, distance='weight'), 0.95))

print('')

print("Highest betweenness centrality", highest_vals(nx.betweenness_centrality(G), 0.95))
# nodes with degree > 3 meaning they are on multiple bus lines



for node in G_nodes_list:

    if(G.degree[node])>3:

        print(node)

        print(G.degree[node])

        print(G.nodes[node])
# printing nodes with high closeness centrality



for n in highest_vals(nx.closeness_centrality(G, distance='weight'), 0.95):

    print(n)

    avg_distance_single(G, n, G_nodes_list)

    print(G.nodes[n])

    print('')
# printing  nodes with high betweenness centrality

bc = nx.betweenness_centrality(G, normalized=False)

for n in highest_vals(bc, 0.95):

    print(n)

    print(bc[n])

    print(G.nodes[n])

    print('')


def draw_cent(G, pos, measures, measure_name):

    

    nodes = nx.draw_networkx_nodes(G, pos, node_size=100, cmap=plt.cm.plasma, 

                                   node_color=list(measures.values()),

                                   nodelist=measures.keys())

    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))

    



    edges = nx.draw_networkx_edges(G, pos)



    plt.title(measure_name)

    plt.colorbar(nodes)

    plt.axis('off')

    plt.show()

    

draw_cent(G, pos, nx.degree_centrality(G), 'Degree Centrality')

draw_cent(G, pos, nx.closeness_centrality(G), 'Closeness Centrality')

draw_cent(G, pos, nx.betweenness_centrality(G), 'Betweenness Centrality')
# closess centrality and betweenness centrality of a given line



def draw_cent_line(G, line, measure_name):

    

    # Convert to network

    L = nx.Graph()   # main graph with all nodes combined



    # Add nodes

    for line_name, nodes in lines.items():

        if(line_name == line):

            for node in nodes:



                L.add_node(node)  # adding nodes to L



                # adding list of lines that nodes are in as node attribute

                if('line' not in L.nodes[node].keys()):

                    L.nodes[node]['line'] = [line_name]

                else:

                    L.nodes[node]['line'].append(line_name)



            # adding edges with weights and line number    

            for i in range(1, len(nodes)):

                L.add_edge(nodes[i - 1], nodes[i], weight=line_distances[line_name][i-1], 

                           line=line_name)

                

    deg_list = []

    for node in G.nodes:

        if(line in G.nodes[node]['line']):

            deg_list.append(G.degree[node]*50)

            

  

    

    pos = nx.spectral_layout(L)

    

    if(measure_name == 'Edge Distance Based Closeness Centrality'):

        measures = nx.closeness_centrality(G, distance='weight')

    elif(measure_name == 'Betweenness Centrality'):

        measures = nx.betweenness_centrality(G)

           

    values = []

    for node, v in measures.items():

        if(line in G.nodes[node]['line']):

            values.append(v)

    

    nodes = nx.draw_networkx_nodes(L, pos, node_size=deg_list, cmap=plt.cm.plasma, 

                                   node_color=values, nodelist=L.nodes)

    



    edges = nx.draw_networkx_edges(L, pos)



    plt.title(measure_name + ' of ' + line.lower())

    plt.colorbar(nodes)

    plt.axis('off')

    plt.show()

    



draw_cent_line(G, "LINE 1", 'Edge Distance Based Closeness Centrality')

draw_cent_line(G, "LINE 1", 'Betweenness Centrality')
k_comps = nxa.k_components(G)

print(len(k_comps))



print("Num of 1-components: " + str(len(k_comps[1])))

print(len(k_comps[1][0]) / len(G.nodes))



print("Num of 2-components: " + str(len(k_comps[2])))

print("length of 2-component: " + str(len(k_comps[2][0])))

print(len(k_comps[2][0]) / len(G.nodes))

print(k_comps[2][0])
def draw_subset_nodes(G, pos, subset, title):

    color = list(map(lambda n: 1 if n in subset else 0, G.nodes))

    nodes = nx.draw_networkx_nodes(G, pos, node_size=100, cmap=plt.cm.inferno, 

                                       node_color=list(color),

                                       nodelist=G.nodes)

    # nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))



    edges = nx.draw_networkx_edges(G, pos)



    plt.title(title)

    plt.axis('off')

    plt.show()



draw_subset_nodes(G, pos, k_comps[2][0], "2-components")

# compute Molloy Reed parameter



num_deg = [0,0,0,0,0,0]

for node in G_nodes_list:

    for i in range(1,7):

        if(G.degree[node]) == i:

            num_deg[i-1] += 1



total_deg=0

total_deg_sq=0

i = 0

for num in num_deg:

    i += 1

    total_deg += i*num

    total_deg_sq += i*i*num

   

avg_deg = total_deg/206

print("The average degree is {:.3f}".format(avg_deg))



sec_moment_k = total_deg_sq/206

M_R_param = sec_moment_k/avg_deg

print("The Molloy Reed parameter is {:.3f}".format(M_R_param))