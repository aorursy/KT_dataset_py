import pandas as pd
import spacy
import networkx as nx
from itertools import combinations
from collections import defaultdict
import operator
import matplotlib.pyplot as plt
import numpy as np
# from math import log
def coocurrence(*inputs):
    com = defaultdict(int)
    
    for named_entities in inputs:
        # Build co-occurrence matrix
        for w1, w2 in combinations(sorted(named_entities), 2):
            com[w1, w2] += 1
            com[w2, w1] += 1  #Including both directions

    result = defaultdict(dict)
    for (w1, w2), count in com.items():
        if w1 != w2:
            result[w1][w2] = {'weight': count}
    return result
# Example coocurrence.
# Originally d is not a key here (since included in previous coocurrences)
# Altered to include ALL now, seems to make difference for my methodology below

coocurrence('abcddc', 'bddad', 'cdda')
# check out the data
data = pd.read_csv('../input/snopes.csv')
data.head(2)
# Could keep in dataframe format? 
# Can make use of other fields for analysis/graph, since not a huge dataset
# drop duplicate claims (and unneccesary columns?)
data.drop_duplicates(subset='claim', inplace=True)

# remove 'examples' (Some odd artifacts that messed with analysis)
data = data.replace({'Example\(s\)': ''}, regex=True)
data = data.replace({'\s+': ' '}, regex=True)
# remove duplicate claims (Not really needed since dropped already)
claims = data.claim.unique()

# make sure it's all strings 
# added lower and whitespace strip just in case
# claims = [str(claim).lower().strip() for claim in claims]
# Turns out this ruins it... and reduced most docs to few claims for some reason

# NER list we'll use - Perhaps could be expanded?
nlp = spacy.load('en_core_web_sm')

# intialize claim counter & lists for our entities
coocur_edges = {}

print('Number of claims: ', len(claims))
# Lets look at the first few claims, along with the ents identified

for doc in nlp.pipe(claims[:5]):
    print(doc)
    print(list(doc.ents))
    print('\n')
# Separating this lengthy step, and saving result as a list rather than generator
# (Size isnt too big, and saves a lot of time when reused later)

# Spacy seems to have error at 3k doc mark? 
# Related to this maybe? https://github.com/explosion/spaCy/issues/1927
# Continuing on with the first 3000 of 3122 for now

corpus = list(nlp.pipe(claims[:3000]))
# Looking at number of times each ent appears in the total corpus
# nb. ents all appear as Spacy tokens, hence needing to cast as str

all_ents = defaultdict(int)

for i, doc in enumerate(corpus):
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

ents_in_claim = [len(doc.ents) for doc in corpus]

plt.hist(ents_in_claim, 
         rwidth=0.9, 
         bins=np.arange(max(ents_in_claim)+2)-0.5)  
        # Futzing with bins just to fix column alignment - not really necessary
plt.title('Entities per claim')
plt.show()
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
# claim_ents = [list(map(str, doc.ents)) for doc in nlp.pipe(claims[:10]*2)]
# Can filter out claims with only 1 ent (nothing to coocur with)

multi_ent_claims = [c for c in claim_ents if len(c)>1]
# single_ent_claims = [c for c in claim_ents if len(c)==1]
# no_ent_claims = [c for c in claim_ents if len(c)==0]

multi_ent_claims
# Generating coocurrence dict of dicts

coocur_edges = coocurrence(*multi_ent_claims)
coocur_edges
# Filter out ents with <2 weight - refactored into a function later.
# (Could also use: del coocur_edges[k1][k2] rather than make new dict)

coocur_edges_filtered = defaultdict()

for k1, e in coocur_edges.items():
    ents_over_2_weight = {k2: v for k2, v in e.items() if v['weight'] >= 1}
    if ents_over_2_weight:  # ie. Not empty
        coocur_edges_filtered[k1] = ents_over_2_weight

coocur_edges_filtered
# Summing all coocurrences in order to see most coocurring edges

coocur_sum = defaultdict(int)
for k1, e in coocur_edges_filtered.items():
    for k2, v in e.items():
        coocur_sum[k1] += v['weight']

sorted_coocur = sorted(coocur_sum.items(), key=operator.itemgetter(1), reverse=True)
sorted_coocur
# Making the list of claims
claim_ents = []
for doc in corpus:
    string_ents = list(map(str, doc.ents))
    claim_ents.append(string_ents)
    
    
# Keeping only claims with multiple entities
multi_ent_claims = [c for c in claim_ents if len(c)>1]
# single_ent_claims = [c for c in claim_ents if len(c)==1]
# no_ent_claims = [c for c in claim_ents if len(c)==0]


# Creating the coocurrance dict
coocur_edges = coocurrence(*multi_ent_claims)
# Filter out ents with < min_weight - useful for graph clarity?

def filter_ents_by_min_weight(edges, min_weight):
    coocur_edges_filtered = defaultdict()
    for k1, e in edges.items():
        ents_over_x_weight = {k2: v for k2, v in e.items() if v['weight'] > min_weight}
        if ents_over_x_weight:  # ie. Not empty
            coocur_edges_filtered[k1] = ents_over_x_weight
    return coocur_edges_filtered
# Looking at the most coocurring edges

filtered_edges = filter_ents_by_min_weight(coocur_edges, 2)

coocur_sum = defaultdict(int)
for k1, e in filtered_edges.items():
    for k2, v in e.items():
        coocur_sum[k1] += v['weight']

sorted_coocur = sorted(coocur_sum.items(), key=operator.itemgetter(1), reverse=True)
print('Most frequent CO-ocurring entity:')
sorted_coocur[:20]
# Getting the data - eg top 30, including only ents with min weight 2
top_n = 30
min_weight = 2
figsize = (20, 15)
scale_nodes = lambda x: (x * 30) + 1
scale_edges = lambda x: 15 * x

filtered_edges = filter_ents_by_min_weight(coocur_edges, min_weight)

top_cooccur = [x[0] for x in sorted_coocur[:top_n]]  
graph_edges = {k:filtered_edges[k] for k in top_cooccur}

# Attempting to graph these top coocurrances
G = nx.from_dict_of_dicts(graph_edges)
pos = nx.kamada_kawai_layout(G)
# pos = nx.circular_layout(G)
# pos = nx.spring_layout(G)
# pos = nx.fruchterman_reingold_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.shell_layout(G)

# Normalise, then scale the line weights
weights = [G[u][v]['weight'] for u, v in G.edges() if u != v]
weights = list(map(lambda x: (x - min(weights)) / (max(weights) - min(weights)), weights))
weights = list(map(scale_edges, weights))

# Scale node weights 
sum_weights = [coocur_sum[n] if coocur_sum[n]>0 else 1 for n in G.nodes]
sum_weights = list(map(scale_nodes, sum_weights))
# sum_weights = list(map(lambda x: 100*log(x), sum_weights))


plt.figure(figsize=figsize)

# nx.draw(G, pos)
nx.draw_networkx_edges(G, pos, alpha=0.2, width=weights)
nx.draw_networkx_nodes(G, pos, alpha=0.2, node_size=sum_weights)
nx.draw_networkx_labels(G, pos)

plt.xticks([])
plt.yticks([])

plt.title('Top coocurrances of named entities in Snopes claims')
plt.show()
# Colour based on average truthiness of claims including that entity? Some sort of clustering?
# Might make sense to keep in dataframe format to make use of other info columns
# No doubt I'm way off track, but hopefully something in here can help :)
# Perhaps some sort of Sparse Matrix (rather than dict of dict) could have been another choice for representing this data?
# Looking forward to next stream!
# A fun little wordcloud picture for the project
# Just using all the named entities without considering coocurrence
from wordcloud import WordCloud
from collections import Counter
from PIL import Image

# Join all the tweet entities into one list
word_list = [j for i in claim_ents for j in i]

# Count occurences of each entity 
word_count_dict=Counter(word_list)

# Make the wordcloud - can use a black/white image for shape mask
#mask = np.array(Image.open("twitter_logo_bw.png"))
wordcloud = WordCloud(background_color="white", max_words=100, 
                      width = 600, height = 300,
                      #mask=mask, contour_width=5, contour_color="skyblue"
                     )
                      
wordcloud.generate_from_frequencies(word_count_dict)

# Show the wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Or save the file
# plt.savefig('yourfile.png', bbox_inches='tight')
# plt.close()
