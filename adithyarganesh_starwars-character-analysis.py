import os

import json

import random

import networkx as nx

import matplotlib.pyplot as plt

from collections import defaultdict

import networkx.algorithms.centrality as centrality

import networkx.algorithms.shortest_paths.generic as nxpath 
def get_sides():

    return {

      "R2-D2": "1",

      "CHEWBACCA": "1",

      "BB-8": "1",

      "QUI-GON": "1",

      "NUTE GUNRAY": "2",

      "PK-4": "1",

      "TC-14": "1",

      "OBI-WAN": "1",

      "DOFINE": "2",

      "RUNE": "2",

      "TEY HOW": "2",

      "EMPEROR": "2",

      "CAPTAIN PANAKA": "1",

      "SIO BIBBLE": "1",

      "JAR JAR": "1",

      "TARPALS": "2",

      "BOSS NASS": "2",

      "PADME": "1",

      "RIC OLIE": "1",

      "WATTO": "2",

      "ANAKIN": "1",

      "SEBULBA": "1",

      "JIRA": "1",

      "SHMI": "1",

      "C-3PO": "1",

      "DARTH MAUL": "2",

      "KITSTER": "1",

      "WALD": "1",

      "FODE/BEED": "1",

      "JABBA": "1",

      "GREEDO": "2",

      "VALORUM": "2",

      "MACE WINDU": "1",

      "KI-ADI-MUNDI": "1",

      "YODA": "1",

      "RABE": "2",

      "BAIL ORGANA": "1",

      "GENERAL CEEL": "1",

      "BRAVO TWO": "1",

      "BRAVO THREE": "1",

      "CAPTAIN TYPHO": "1",

      "SENATOR ASK AAK": "2",

      "ORN FREE TAA": "2",

      "SOLA": "1",

      "JOBAL": "1",

      "RUWEE": "1",

      "TAUN WE": "1",

      "LAMA SU": "1",

      "BOBA FETT": "2",

      "JANGO FETT": "2",

      "OWEN": "1",

      "BERU": "1",

      "CLIEGG": "1",

      "COUNT DOOKU": "2",

      "SUN RIT": "2",

      "POGGLE": "2",

      "PLO KOON": "1",

      "ODD BALL": "1",

      "GENERAL GRIEVOUS": "2",

      "FANG ZAR": "1",

      "MON MOTHMA": "1",

      "GIDDEAN DANU": "1",

      "CLONE COMMANDER GREE": "1",

      "CLONE COMMANDER CODY": "1",

      "TION MEDON": "1",

      "CAPTAIN ANTILLES": "1",

      "DARTH VADER": "2",

      "LUKE": "1",

      "CAMIE": "1",

      "BIGGS": "1",

      "LEIA": "1",

      "MOTTI": "2",

      "TARKIN": "2",

      "HAN": "1",

      "DODONNA": "1",

      "GOLD LEADER": "1",

      "WEDGE": "1",

      "RED LEADER": "1",

      "RED TEN": "1",

      "GOLD FIVE": "1",

      "RIEEKAN": "1",

      "DERLIN": "1",

      "ZEV": "1",

      "PIETT": "2",

      "OZZEL": "2",

      "DACK": "1",

      "JANSON": "1",

      "NEEDA": "1",

      "LANDO": "1",

      "JERJERROD": "2",

      "BIB FORTUNA": "2",

      "BOUSHH": "1",

      "ADMIRAL ACKBAR": "1",

      "LOR SAN TEKKA": "1",

      "POE": "1",

      "KYLO REN": "2",

      "CAPTAIN PHASMA": "2",

      "FINN": "1",

      "UNKAR PLUTT": "1",

      "REY": "1",

      "GENERAL HUX": "2",

      "LIEUTENANT MITAKA": "1",

      "BALA-TIK": "1",

      "SNOKE": "2",

      "MAZ": "1",

      "SNAP": "1",

      "ADMIRAL STATURA": "1",

      "YOLO ZIFF": "1",

      "COLONEL DATOO": "1",

      "ELLO ASTY": "1",

      "JESS": "1",

      "NIV LEK": "1"

}
def get_value(name, maps):

    for nodes in maps:

        if nodes["name"] == name:

            return nodes["value"]

        

def sort(hashmap):

    return sorted(hashmap.items(), key = lambda x: x[1])[::-1]



def get_info(data):

    connections = defaultdict(int)

    interactions = defaultdict(int)

    for link in data["links"]:

        connections[data["nodes"][link["source"]]["name"]] += 1

        connections[data["nodes"][link["target"]]["name"]] += 1

        interactions[data["nodes"][link["source"]]["name"]] += link["value"]

        interactions[data["nodes"][link["target"]]["name"]] += link["value"]

    return connections, interactions

   

def get_graph(data, episode):

    nodes = [ node['name'] for node in data["nodes"] ]

    edges = [ (nodes[link['source']], nodes[link['target']]) for link in data["links"] ]

    G = nx.Graph()

    for node in nodes:

        G.add_node(node)

    for edge in edges:

        G.add_edge(edge[0],edge[1])

        

#     Uncomment block for graph visualization



    plt.figure(figsize=(30,15))

    plt.subplot(121)

    nx.draw(G, with_labels=True)

    plt.savefig(f'task4_results/graphs/Episode_{episode}_Graph.png')

    plt.show()

    plt.close()



    return G
def get_homophily(data, characters):

    sides = []

    for side in characters.values():

        arr = []

        for count in range(len(data["nodes"])):

            if data["nodes"][count]["name"] in side:

                arr.append(count)

        sides.append(arr)

    

    homophily = []

    for side in sides:

        interacts = defaultdict(int)

        for link in data["links"]:

            if link["source"] in side:

                interacts[data["nodes"][link["target"]]["name"]] += link["value"]

                interacts[data["nodes"][link["source"]]["name"]] += link["value"]

            elif link["target"] in side:

                interacts[data["nodes"][link["source"]]["name"]] += link["value"]

                interacts[data["nodes"][link["target"]]["name"]] += link["value"]

        homophily.append(sorted(interacts.items(), key = lambda x: x[1])[::-1][:10])

    return homophily

                

def light_dark_classification(data, sides):

    light_side = []

    dark_side = []

    for i,j in sides.items():

        if j == '1':

            light_side.append(i)

        else:

            dark_side.append(i)



    sides = []

    for side in [light_side, dark_side]:

        arr = []

        for count in range(len(data["nodes"])):

            if data["nodes"][count]["name"] in side:

                arr.append(count)

        sides.append(arr)



    good = defaultdict(int)

    bad = defaultdict(int)

    for character in sides[0]:

        for link in data["links"]:

            if (link["source"] == character and link["target"] in sides[0]) or (link["target"] == character and link["source"] in sides[0]):

                good[data["nodes"][character]["name"]] += link["value"]/len(sides[0])  

            elif (link["source"] == character and link["target"] in sides[1]) or (link["target"] == character and link["source"] in sides[1]):

                bad[data["nodes"][character]["name"]] += link["value"]/len(sides[1])



    for elem in good.keys():

        if elem not in bad.keys():

            bad[elem] = 0



    for elem in bad.keys():

        if elem not in good.keys():

            good[elem] = 0



    correct = 0

    incorrect = 0

    for l, d in zip(sorted(good.items()), sorted(bad.items())):

        if l[1] >= d[1]:

            if l[0] in light_side:

                correct += 1

            else:

                incorrect += 1

        else:

            if d[0] in dark_side:

                correct += 1

            else:

                incorrect += 1



    return (correct, incorrect)
def loop(episode, feature):

    

    e = "episode-" + str(episode)

    

    if episode == 0:

        e = "full"

    

    with open(f'../input/star-wars/starwars-{e}-{feature}.json') as f:

        data = json.load(f)

        

    sides = get_sides()

    

    characters = {

        "Light Side": ["FINN", "OBI-WAN", "YODA", "PADME", "LUKE"], 

        "Dark Side": ["EMPEROR","DARTH VADER", "PIETT", "GENERAL HUX", "NUTE GUNRAY"]

    }    

    

    probability = 0.7

    

#     Task 2 Hypothesis 1

    connections, interractions = get_info(data)

    homophily = get_homophily(data, characters)

    classification = light_dark_classification(data, sides)

    

    Graph  = get_graph(data, episode)

    

    #     Task 2 Hypothesis 2

    betweenness = sort(centrality.betweenness_centrality(Graph))[:5] + sort(centrality.betweenness_centrality(Graph))[-5:]

    degree_centrality = sort(connections)[:5] + sort(connections)[-5:]



    #     Task 3

    cliquishness = sort(nx.clustering(Graph))[:5] + sort(nx.clustering(Graph))[-5:]

    path_length = list(nxpath.shortest_path_length(Graph))

    randomness = get_randomness(Graph, probability)

    

#     Uncomment to display randomness graphs and save them

#     for graph in range(1, len(randomness[0]), 10):

#         plt.figure(figsize=(30,15))

#         plt.subplot(121)

#         nx.draw(randomness[0][graph], with_labels=True)

#         plt.savefig(f'task3_results/Episode{episode}_{graph}.png')

#         plt.close()

        

#     Task 4, 5

    temp = set()

    for i,j in zip(connections.items(), interractions.items()):

        temp.add((i[0], i[1], j[1], get_value(i[0], data["nodes"])))

    temp = sorted(temp, key = lambda x: x[1])[::-1]

    

#     Uncomment block for visualizations



    plt.figure(figsize=(25,10))

    plt.title(f'Episode-{episode} {feature}')

    plt.plot(list(zip(*temp))[0], list(zip(*temp))[1], list(zip(*temp))[0], list(zip(*temp))[2], list(zip(*temp))[0], list(zip(*temp))[3])

    plt.xticks(list(zip(*temp))[0][::1],  rotation='vertical')

    plt.savefig(f'task4_results/images/Episode_{episode}_{feature}.png')

    plt.show()

    plt.close()

    

    return (homophily, classification), (betweenness, degree_centrality), (cliquishness, path_length, randomness[1]), (connections, interractions)

def get_randomness(G, p):

    G.remove_nodes_from(list(nx.isolates(G)))

    spread = []

    spread_paths = []

    for j in range(1, len(list(G))):

        spread.append(G)

        spread_paths.append(nx.average_shortest_path_length(G))

        if random.randint(0, 100) < p*100:

            edges = list(set(G.edges) - set(nx.bridges(G)))

            if edges:

                u, v = random.choice(edges)

                G.remove_edge(u, v)

                w = random.choice(list(set(G) - set(x for _,x in set(G.edges(u))))) 

                G.add_edge(u, w)

    return spread, spread_paths
if __name__ == "__main__":



    if not os.path.exists('task3_results'):

        os.makedirs('task3_results')

    if not os.path.exists('task4_results'):

        os.makedirs('task4_results')

        os.makedirs('task4_results/images')

        os.makedirs('task4_results/graphs')

        

#     Define feature depending on what characteristics you wish to amalyze with

    feature = "interactions-allCharacters"

#     features = "mentions"



    # considering episode 0 as full series

    for episode in range(0,8):

        hypothesis_1, hypothesis_2, task_3, task_4_and_5 = loop(episode, feature)

        print(f"\n Episode {episode} \n\n Task 1 and 2")

        print(f"\n Hypothesis 1 Analysis \n\n Homophily \n {hypothesis_1[0]} \n\n Classification (correct side, wrong side) {hypothesis_1[1]}")

        print(f"\n Hypothesis 2 Analysis \n\n Betweenness \n {hypothesis_2[0]} \n\n Degree_centrality \n {hypothesis_2[1]}")        

        print(f"\n Task 3 \n\n Cliquishness \n {task_3[0]} \n\n Path_length \n {task_3[1][0]} \n \n Average Path Length when adding randomness\n {task_3[2]}")

        print(f"\n Task 4 and 5 \n\n Connections \n {task_4_and_5[0]} \n\n Interactions \n {task_4_and_5[1]}")

# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the read-only "../input/" directory

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session