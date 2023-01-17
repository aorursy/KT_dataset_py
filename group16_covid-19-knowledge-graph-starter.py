!pip install rdflib

!apt-get -y install python-dev graphviz libgraphviz-dev pkg-config

!pip install pygraphviz
import rdflib

import networkx as nx

%matplotlib inline

import matplotlib.pyplot as plt
# This takes a while...

g = rdflib.Graph()

g.parse('/kaggle/input/covid19-literature-knowledge-graph/kg.nt', format='nt')
# Number of triples

print(len(list(g.triples((None, None, None)))))



#Ppredicates

print(len(set(g.predicates())))



# Number of subjects

print(len(set(g.subjects())))
rand_paper = rdflib.URIRef('http://dx.doi.org/10.1016/j.mbs.2013.08.014')

for i, (s, p, o) in enumerate(g.triples((rand_paper, None, None))):

    print(s, p, o)
for s, p, o in g.triples((rand_paper, None, None)):

    print(s, p, o)
preds = set()

for s, p, o in g.triples((rdflib.URIRef('http://idlab.github.io/covid19#bf20dda99538a594eafc258553634fd9195104cb'), None, None)):

    print(s, p, o)
def create_sub_graph(root, depth):

    # Limit number of hasWords relations to not overcrowd the figure

    words_cntr = 0

    

    # Get all the triples that are maximally 2 hops away from our randomly picked paper

    objects = set()

    nx_graph = nx.DiGraph()

    

    rdf_subgraph = rdflib.Graph()

    to_explore = {root}

    for _ in range(depth):

        new_explore = set()

        for node in to_explore:

            for s, p, o in g.triples((node, None, None)):

                if 'words' in str(p).lower():

                    if words_cntr >= 25:

                        continue

                    words_cntr += 1



                s_name = str(s).split('/')[-1][:25]

                p_name = str(p).split('/')[-1][:25]

                o_name = str(o).split('/')[-1][:25]

                nx_graph.add_node(s_name, name=s_name)

                nx_graph.add_node(o_name, name=o_name)

                nx_graph.add_edge(s_name, o_name, name=p_name)

                rdf_subgraph.add((s, p, o))

                

                new_explore.add(o)

        to_explore = new_explore

    return nx_graph, rdf_subgraph

    

nx_graph, rdf_subgraph = create_sub_graph(rand_paper, 3)

        

plt.figure(figsize=(20, 20))

_pos = nx.kamada_kawai_layout(nx_graph)

_ = nx.draw_networkx_nodes(nx_graph, pos=_pos)

_ = nx.draw_networkx_edges(nx_graph, pos=_pos)

_ = nx.draw_networkx_labels(nx_graph, pos=_pos, fontsize=8)

names = nx.get_edge_attributes(nx_graph, 'name')

_ = nx.draw_networkx_edge_labels(nx_graph, pos=_pos, edge_labels=names, fontsize=8)
rdf_subgraph.serialize(destination='sub.ttl', format='turtle')