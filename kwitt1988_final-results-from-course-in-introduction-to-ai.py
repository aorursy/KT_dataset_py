!pip install apyori
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import networkx as nx

import itertools as it

import matplotlib.pyplot as plt

import apyori as ap



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/food-com-recipes-and-user-interactions/RAW_recipes.csv', usecols=[0, 10])

G = nx.Graph()



currentRecipie = 1

while currentRecipie < 1000:

    currentRecipeIngredients = df.at[currentRecipie, 'ingredients'].replace("\'", '').replace("[", "").replace("]", "").replace(" ", "").split(',') 



    dy = pd.DataFrame(list(it.combinations(currentRecipeIngredients, 2)), columns=["x", "y"])

    dx = pd.DataFrame(currentRecipeIngredients, columns=["ingredients"])

    

    # DX = CandidateSet?



    for index, row in dx.iterrows():

        if row['ingredients'] in list(G.nodes):

            x = row['ingredients']

            y = G.nodes[x]['weight'] + 1

            G.add_node(row['ingredients'], weight=y)

        else:

            G.add_node(row['ingredients'], weight=1)

        

    for index, row in dy.iterrows():

        if G.has_edge(row['y'], row['x']):

            x = [row['x'], row['y']]

            y = G.edges[x]['weight'] + 1

            G.add_edge(row['x'], row['y'], weight=y)

        else:

            G.add_edge(row['x'], row['y'], weight=1)



    currentRecipie += 1

    

print(G.number_of_nodes())
for (u, v) in G.edges:

    if G.edges[u, v]['weight'] <= 10:

        G.remove_edge(u, v)



for u in list(G.nodes):

    if G.nodes[u]['weight'] <= 10:

        G.remove_node(u)



G.remove_nodes_from(list(nx.isolates(G)))



plt.figure(figsize=(18,18))

nx.draw_spring(G, with_labels=True)

plt.show()

plt.savefig("plot.png", dpi=500)