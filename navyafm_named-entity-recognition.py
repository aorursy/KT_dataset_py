

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy



data=pd.read_csv("../input/rumor-citation/snopes.csv")
data.head()
claims = data.claim.unique()
from  itertools import combinations

from collections import defaultdict



def cooccurence(common_entities):

    com =  defaultdict(int)

    for w1,w2 in combinations(sorted(common_entities),2):

        com[w1, w2] += 1

    result =  defaultdict(dict)

    for (w1,w2), count in com.items():

        if w1!= w2:

            result[w1][w2]= {'weight':count}

    return result


from itertools import combinations 



nlp = spacy.load("en_core_web_sm")

#docs = nlp.pipe(claims)



cooccur_edges = {}

claims = [str(claim) for claim in claims]

for docs in nlp.pipe(claims,n_threads=3):

    cooccur_edges.update(cooccurence(docs.ents))
cooccur_edges
#to plot Networkx from Dataframe.

import networkx as nx

G=nx.from_dict_of_dicts(cooccur_edges)

nx.draw(G)