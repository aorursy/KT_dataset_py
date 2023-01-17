# Install pygithub and networkx

!pip install pygithub

!pip install networkx
# Importing packages

from github.MainClass import Github

import networkx as nx

import json

import matplotlib.pyplot as plt
# function to generate graph

def mount_graph(name_repo, graph):

    nx.set_node_attributes(graph,'qt_contr',1)

    list_repo = [name_repo]

    g = Github('5ce7ddafe7c90a6fa6254a9041313698830b6c64')

    with open('../input/countries.json', 'r') as f:

        data=f.read()

    paises = json.loads(data)

    try:

        repo =g.get_repo(name_repo)

        i = 0

        graph.add_node(repo.name)

        graph.nodes[repo.name]["qt_contr"] = 0

        for user in repo.get_contributors():

            i=i+1

            if(i > 500):

                break

            graph.nodes[repo.name]["qt_contr"] = graph.nodes[repo.name]["qt_contr"] +1

            if(user.location is not None):

                place = user.location.replace(" ","").split(",") 

                for pla in place:

                    if(pla in paises.keys() ):

                        if(graph.has_node(pla)):

                            graph.nodes[pla]["qt_contr"] = graph.nodes[pla]["qt_contr"] +1

                            break

                        else:

                            graph.add_node(pla)

                            graph.nodes[pla]["qt_contr"] = 1

                            graph.add_edge(repo.name,pla)

                            break

                    else:

                        for key,value in paises.items():

                            if(pla in paises[key]):

                                if(graph.has_node(key)):

                                    graph.nodes[key]["qt_contr"] = graph.nodes[key]["qt_contr"] +1

                                else:

                                    graph.add_node(key)

                                    graph.nodes[key]["qt_contr"] = 1

                                    graph.add_edge(repo.name,key)

        return graph

    except Exception as ex:

        print(ex)
# function to plot and save graph

def plot_save(repo_name):

    name = repo_name.split("/")[-1]

    plt.figure(figsize = (30,30))

    graph = nx.Graph()

    graph = mount_graph(repo_name, graph)

    pos = nx.spring_layout(graph, scale=3)

    n = graph.number_of_nodes()

    nx.draw(graph, pos, node_size=40000, node_color=range(n), with_labels=True, font_size=20, font_weight='bold', cmap=plt.cm.Blues)

    nx.write_graphml(graph, name+"_countries.graphml")
# python repo

plot_save("python/cpython")
# swift repo

plot_save("apple/swift")
# tensorflow repo

plot_save("tensorflow/tensorflow")
# qiskit repo

plot_save("Qiskit/qiskit")