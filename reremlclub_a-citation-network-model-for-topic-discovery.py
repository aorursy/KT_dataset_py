from IPython.display import Image

Image(filename='/kaggle/input/citationnetworkfigure/citationNetwork.png', width=400)
!pip install git+https://github.com/ReReMLclub/cord19utils.git/ --quiet
import numpy as np 

import pandas as pd 

import glob

import json

from collections import defaultdict

import cord19utils

import holoviews as hv

import networkx as nx
hv.extension('bokeh')

hv.output(size=300)
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})
filePaths = glob.glob(f'{root_path}/**/*.json', recursive=True)
reader = cord19utils.CorpusReader(filePaths, meta_df)
builder = cord19utils.GraphBuilder(reader)
graph = builder.buildGraph(citeOutCutoff = 10, citeInCutoff = 50, weightBound = 3)

print(f'Number of nodes: {graph.number_of_nodes()}\nNumber of edges: {graph.number_of_edges()}')
communities = builder.assignCommunities(graph, nCommunities = 25)
proc = cord19utils.TextProcessor(graph)

proc.buildDictionary()

proc.assignCommunityTopics(communities, verbose = True)
sgraph = builder.buildSupergraph(graph, communities, weightCutoff = 200)
cord19utils.drawChordGraph(sgraph, proc.id2label)
for node, label in proc.id2label.items():

    if 'asthma' in label: nodeOfInterest = node
cord19utils.drawChordGraph(sgraph, proc.id2label, nodeOfInterest = nodeOfInterest)