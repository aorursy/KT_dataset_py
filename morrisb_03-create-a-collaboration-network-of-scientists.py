# To store the data

import pandas as pd



# To create networks

import networkx as nx



# To get efficient counters

from collections import Counter



# To get all possible permutations

from itertools import permutations
########## Load The Data ##########

# Load the csv file

df = pd.read_csv('../input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')



# Drop all duplicated abstracts to have each publication only a single time

df = df.drop_duplicates(subset=['abstract'])



# Get the authors and drop empty values

df = df['authors'].dropna()











########## Extrat All Pairs Of Collaborations ##########

# List to store the collaboration pairs

collaboration_list = []



# Iterate over all authors

for authors in df.values:



    # Split the authors into a list

    authors_list = authors.split('; ')

    

    # Iterate over all possible collaboration pairs (every pair will appear twice!)

    for collaboration in permutations(authors_list, 2):

        

        # Sort the pair alphabetical

        collaboration = sorted(collaboration)

        

        # Store the collaboration pair (to use the Counter the sorted list has to be converted to string)

        collaboration_list.append('; '.join(collaboration))











########## Count And Sort Collaborations / Edges ##########

# Count the collaborations for the network weights

counter = Counter(collaboration_list)



# Split the collaborations again and account for the double count of collaborations

counter = [[key.split('; '), value/2] for key, value in counter.items()]
########## Create The Graph ##########

# Create a graph

graph = nx.Graph()



# Iterate over all collaborations / edges

for (v, w), weight in counter:

    

    # Add weighted edges to the graph

    graph.add_edge(v, w, weight=weight)





print('The graph as {} nodes.'.format(len(graph.nodes)))

print('The graph as {} edges.'.format(len(graph.edges)))











########## Compute The Authors PageRank ##########

# Compute the PageRank for each node of the network

pagerank = nx.pagerank(graph, weight='weight')



# Sort the dictionary entries by its values

pagerank_sorted = [[k, v] for k, v in sorted(pagerank.items(), key=lambda item: item[1], reverse=True)]



print('\n\nPageRank results for the 20 most important authors weighted by their number of publications:\n')

for name, rank in pagerank_sorted[:20]:

    print('{}:\t\t{}'.format(name, rank))