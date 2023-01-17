# Having a play with NER method as seen on Youtube - https://www.youtube.com/watch?v=4SoLoo0fdH0
# Thanks to Rachael Tatman
import pandas as pd
import spacy
import networkx as nx
from itertools import combinations
from collections import defaultdict
import operator
import matplotlib.pyplot as plt
import numpy as np
from math import log
data.claim[0]
# check out the data
data = pd.read_csv("../input/snopes.csv")
data.head()

# remove examples
data = data.replace({'Example\(s\)': ''}, regex=True)
data = data.replace({'\s+': ' '}, regex=True)
# Method where you pass in a list of lists of ents, outputting coocur dict
def coocurrence(*inputs):
    com = defaultdict(int)
    
    for common_entities in inputs:
        # Build co-occurrence matrix
        for w1, w2 in combinations(sorted(common_entities), 2):
            com[w1, w2] += 1

    result = defaultdict(dict)
    for (w1, w2), count in com.items():
        if w1 != w2:
            result[w1][w2] = {'weight': count}    
    return result
# test coocurrence
# Why is d not a key here? Intended? Saving on redundancy? 

coocurrence('abcddc', 'bddad', 'cdda')
# remove duplicate claims
claims = data.claim.unique()

# make sure it's all strings 
# added lower and whitespace strip just in case
# claims = [str(claim).lower().strip() for claim in claims]
# Turns out this ruins it... and reduced most docs to few claims for some reason

# NER list we'll use - Maybe this needs looking at?
nlp = spacy.load('en_core_web_sm')

# intialize claim counter & lists for our entities
coocur_edges = {}

print('Number of claims: ', len(claims))
# Working much better without the str casting now. 
# Redoing whole sheet from here

for doc in nlp.pipe(claims[:10]):
    print(doc)
    print(list(doc.ents))
# Looking at number of times each ent appears in the total corpus
# nb. ents all appear as Spacy tokens, hence needing to cast as str for dict

# Spacy seems to have error at 3k doc mark? 
# Related to this maybe? https://github.com/explosion/spaCy/issues/1927
# Continuing on with the first 3000 of 3122 for now

all_ents = defaultdict(int)

for i, doc in enumerate(nlp.pipe(claims[:3000])):
    #print(i,doc)
    for ent in doc.ents:
        all_ents[str(ent)] += 1
        
print('Number of distinct entities: ', len(all_ents))
# Most popular ents
sorted_ents = sorted(all_ents.items(), key=operator.itemgetter(1), reverse=True)
sorted_ents[:20]
# Number of ents that appear at least twice

multi_ents = [x for x in sorted_ents if x[1] > 1]

print('Number of ents that appear at least twice: ', len(multi_ents))
# How many ents appear per claim?
# Blank strings (non breaking spaces?) popular?

ents_in_claim = [len(doc.ents) for doc in nlp.pipe(claims[:3000])]

plt.hist(ents_in_claim, 
         rwidth=0.9, 
         bins=np.arange(max(ents_in_claim)+2)-0.5)  
# Futzing with bins to fix column alignment
plt.title('Entities per claim')
plt.show()
# Exploring/Demonstrating with small subset, then performing calcs with whole dataset afterwards
# Listing claims as a list of their entities

claim_ents = []
for i, doc in enumerate(nlp.pipe(claims[:5])):
    string_ents = list(map(str, doc.ents))
    claim_ents.append(string_ents)
    # Doubling some up to fake/force coocurrence
    if i%2==0:
        claim_ents.append(string_ents)  
claim_ents

# Could do as a one line list comprehension, though maybe not as readable:
# claim_ents = [list(map(str, doc.ents)) for doc in nlp.pipe(claims[:5])]
# Can filter out claims with only 1 ent (nothing to coocur with)

multi_ent_claims = [c for c in claim_ents if len(c)>1]
# single_ent_claims = [c for c in claim_ents if len(c)==1]
# no_ent_claims = [c for c in claim_ents if len(c)==0]

multi_ent_claims
# Generating coocurrence dict of dicts
# Something funny with that /xa0 non breaking space... at least the method seems to be working?

coocur_edges = coocurrence(*multi_ent_claims)
coocur_edges
# Filter out ents with <2 weight
# Could also use: del coocur_edges[k1][k2] rather than make new dict
coocur_edges_filtered = defaultdict()

for k1, e in coocur_edges.items():
    ents_over_2_weight = {k2: v for k2, v in e.items() if v['weight'] > 1}
    if ents_over_2_weight:  # ie. Not empty
        coocur_edges_filtered[k1] = ents_over_2_weight

coocur_edges_filtered
# Summing all coocurrences in order to see most coocurring edges

coocur_sum = defaultdict(int)
for k1, e in coocur_edges_filtered.items():
    for k2, v in e.items():
        coocur_sum[k1] += v['weight']
coocur_sum
# Now to retry with whole dataset. Here goes nothin...
# Making the list of claims
claim_ents = []
for doc in nlp.pipe(claims[:3000]):
    string_ents = list(map(str, doc.ents))
    claim_ents.append(string_ents)
      
# Keeping only claims with multiple entities
multi_ent_claims = [c for c in claim_ents if len(c)>1]
# single_ent_claims = [c for c in claim_ents if len(c)==1]
# no_ent_claims = [c for c in claim_ents if len(c)==0]

# Creating the coocurrance dict
coocur_edges = coocurrence(*multi_ent_claims)
# Filter out ents with < x weight - change this for graph clarity?
coocur_edges_filtered = defaultdict()
for k1, e in coocur_edges.items():
    ents_over_x_weight = {k2: v for k2, v in e.items() if v['weight'] > 3}
    if ents_over_x_weight:  # ie. Not empty
        coocur_edges_filtered[k1] = ents_over_x_weight
        
# Looking at the most coocurring edges
coocur_sum = defaultdict(int)
for k1, e in coocur_edges_filtered.items():
    for k2, v in e.items():
        coocur_sum[k1] += v['weight']

sorted_coocur = sorted(coocur_sum.items(), key=operator.itemgetter(1), reverse=True)
print('Most frequent CO-ocurring entity:')
top_cooccur = sorted_coocur[:20]
top_cooccur
# Getting the data - top10, excl. that space occurring 3k times..
top_cooccur_no_space = [x[0] for x in top_cooccur[:50]]  
graph_edges = {k:coocur_edges_filtered[k] for k in top_cooccur_no_space}

# Attempting to graph these top 10 coocurrances
G = nx.from_dict_of_dicts(graph_edges)
pos = nx.spring_layout(G)

# Normalise, then scale the line weights
weights = [G[u][v]['weight'] for u, v in G.edges() if u != v]
weights = list(map(lambda x: (x - min(weights)) / (max(weights) - min(weights)), weights))
weights = list(map(lambda x: (x * 6) + 1, weights))

# Scale node weights on log scale 
sum_weights = [coocur_sum[n] if coocur_sum[n]>0 else 1 for n in G.nodes]
sum_weights = list(map(lambda x: 10*x, sum_weights))
# sum_weights = list(map(lambda x: 100*log(x), sum_weights))
# '\xa0Example(s' with 0 weight throwing off scaling?


plt.figure(figsize=(20,10))

# nx.draw(G, pos)
nx.draw_networkx_edges(G, pos, alpha=0.2, width=weights)
nx.draw_networkx_nodes(G, pos, alpha=0.2, node_size=sum_weights)
nx.draw_networkx_labels(G, pos)

plt.xticks([])
plt.yticks([])

plt.title('Top coocurrances of named entities in Snopes claims')
plt.show()