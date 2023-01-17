# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Players.csv')

names = data['Player'].str.split(' ',expand=True)

names.columns = ['first', 'last']

names = names.dropna() # Removes Nene

names = names[names['first'] != names['last']] # Removes Ha Ha and Sun Sun



import networkx as nx

G = nx.DiGraph()

G.add_nodes_from(list(names['first']))

G.add_nodes_from(list(names['last']))

#Add edges

G.add_edges_from(list(zip(names['first'],names['last'])))

' '.join(nx.dag_longest_path(G))