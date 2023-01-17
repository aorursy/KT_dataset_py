from   Bio import Entrez

from   Bio import Medline

import matplotlib.pylab as plt

import pandas as pd

import numpy as np

import itertools

import networkx as nx

import time

import collections
# setup your email for PubMed searches

Entrez.email = "youremail@yourisp.org" #if you breach PubMed terms you'll be warned here

# setup a couple of other pubmed search parameters

maxrecords="10000" # please respect the limits in volumes, days of the week or get banned

corpus="pubmed"
# this is a sample MeSH search string. Google for tutorials on how to query PubMed and PMC

searchstring='("Diabetes Mellitus"[Mesh]) AND "Artificial Intelligence"[Mesh]'
handle = Entrez.esearch(db=corpus, term=searchstring, retmax=maxrecords, sort="relevance", retmode="xml")

records = Entrez.read(handle)

ids = records['IdList'] # save the list of article IDs to be used later on in the efetch

print("Found {} articles using the following query string: {}".format(records["Count"],searchstring))
# now fetch the article's data

h = Entrez.efetch(db='pubmed', id=ids, rettype='medline', retmode='text')

# now records hold a dictionary of matching publications

records = Medline.parse(h)
# create the new multigraph. A multigraph can hold N edges between a couple of nodes. It limits some algos

# but in this case I will have one edge for each coauthorship

# we will use authors_dict for the nodes, coa_df for the edges, pub_df for the edges attributes

G = nx.MultiGraph(name=searchstring)
# now let's iterate through the articles, fetch the data and populate the multigraph

for record in records:

    au = record.get('AU', '?') # list of authors

    dp = record.get('DP', '?') # date of publication

    yp = dp.split(' ', 1)[0]  # keep only the first word in the string, the year

    pi = record.get('PMID', '?') # publication ID

    # loop through each paper's author list and add edges to the graph with auth1-auth2 and PMID as attribute

    coauth_pairs_ls = list(itertools.combinations(au, 2))

    for pair in coauth_pairs_ls:

        auth1, auth2 = pair

        G.add_edge(auth1, auth2,pmid=pi,year=yp) 
# the multigraph has been built, let's spew out some general statistics

# print some Graph statistics

print("Some statistics of this graph:")

print(nx.info(G))

print("The density of the graph is {}.".format(nx.density(G)))

gclique = list(nx.find_cliques(G))

print("There are {} cliques in the graph.".format(len(gclique)))

gcomps = nx.connected_components(G)

numcomps = sum(1 for w in gcomps) #gcomps is a generator so I am counting its output in a loop

print("There are {} connected component in the graph.".format(numcomps))
# finally do a primitive drawing of the graph

# this will take some time with the provided medium sized graph of the sample query

nx.draw(G, with_labels=False)