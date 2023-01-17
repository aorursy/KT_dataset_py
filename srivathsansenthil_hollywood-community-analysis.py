# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
credits=pd.read_csv('/kaggle/input/the-movies-dataset/credits.csv')
credits.head()
k=credits['cast'][0]

import ast 

res = ast.literal_eval(k)

print(res[0])
import networkx as nx

MG=nx.MultiGraph()
credits.info()

from itertools import combinations 

def rSubset(arr): 

    return list(combinations(arr,2))
for i in range(45476):

    cast=credits['cast'][i]

    res = ast.literal_eval(cast)

    cast_mem=[]

    for j in range(len(res)):

        cast_mem.append(res[j]['name'])

    edges=rSubset(cast_mem)

    for k in range(len(edges)):

        MG.add_edge(edges[k][0],edges[k][1])
print('no of nodes: ',len(MG.nodes()))
print('no of edges: ',len(MG.edges()))
conn_comp=list(nx.connected_components(MG))
print('no of connected components:',len(conn_comp))
degree_sequence = sorted([len(n) for n in conn_comp], reverse=True) 

print(degree_sequence[:5])

print('The highly connected componenet has no of nodes : ',degree_sequence[0])
#removing componenets with less no of nodes i.e no of nodes<50

for n in conn_comp:

    if len(n)<50:

        MG.remove_nodes_from(n)

conn_comp=list(nx.connected_components(MG))

print('no of connected components:',len(conn_comp))
import collections

import matplotlib.pyplot as plt

degree_sequence = sorted([d for n, d in MG.degree()], reverse=True) 

degreeCount = collections.Counter(degree_sequence)

deg, cnt = zip(*degreeCount.items())

plt.figure(figsize=(20,10))

plt.plot(deg, cnt)

plt.title("Degree Histogram")

plt.ylabel("Count")

plt.xlabel("Degree")

plt.xlim(0,3000)
print("avg degree: ",sum(degree_sequence)/len(MG.nodes()))
#options = {'node_color': 'black','node_size': 50,'width': 1 }

#nx.draw_random(MG, **options)

#plt.show()