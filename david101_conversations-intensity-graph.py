from collections import defaultdict



import numpy as np

import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt



import seaborn as sns

sns.set_style('whitegrid')



characters = pd.read_csv('../input/simpsons_characters.csv', index_col='id')

script_lines = pd.read_csv('../input/simpsons_script_lines.csv', index_col='id', 

                           low_memory=False, usecols=list(range(13)))
episode_ids = script_lines['episode_id'].unique()

DG = nx.DiGraph()

for episode_id in episode_ids:

    episode = script_lines[script_lines['episode_id'] == episode_id]

    current_room, current_character = -1, -1

    previous_room, previous_character = -2, -2

    for line_id, row in episode.sort_values(by="timestamp_in_ms").iterrows():

        if row['speaking_line']:

            try:

                current_room = int(row['location_id'])

            except:

                current_room = -1

                

            try:

                current_character = int(row['character_id'])

            except:

                current_character = -1

                

            if current_room == previous_room:

                if ((previous_character not in DG) or 

                   (current_character not in DG[previous_character])):

                    DG.add_edge(previous_character, current_character, weight=1)

                else:

                    DG[previous_character][current_character]['weight'] += 1



            previous_character = current_character

            previous_room = current_room        
avg_weight = np.mean([DG[e[0]][e[1]]['weight'] for e in DG.edges_iter(data=True)])

max_weight = np.amax([DG[e[0]][e[1]]['weight'] for e in DG.edges_iter(data=True)])

print("Average amount of conversations between characters: {}".format(avg_weight))

print("Max lines of conversations between characters: {}".format(max_weight))

useful_nodes = [e[0] for e in DG.edges_iter(data=True) 

                     if (DG[e[0]][e[1]]['weight'] > avg_weight * 45) 

                        and (e[0] != -1) 

                        and (e[0] != e[1])]

useful_DG = DG.subgraph(useful_nodes)

print("Number of edges between the most talkative characters: {}".format(useful_DG.number_of_edges()))
# here we get names from IDs

labels = {node: characters.loc[node]['name']  for node in useful_DG.nodes()}

edges = useful_DG.edges()

# Line width is proportional to the conversation intensity

weights = [useful_DG[u][v]['weight'] / max_weight * 10 for u, v in edges]

plt.figure(figsize=(10, 9))

# I turned off arrows, so width shows MAXIMUM of 

# (char_1, char_2) and (char_2, char_1) dialogue lines

nx.draw_circular(useful_DG, arrows=False, edges=edges, with_labels=True, node_shape='s',

                 labels=labels, node_color='#FFD90F', width=weights, node_size=1600)

plt.title("Conversations intensity graph")

plt.show()