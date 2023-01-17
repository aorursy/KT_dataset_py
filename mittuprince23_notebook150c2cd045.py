# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas  as pd

import plotly.express as px

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filename='/kaggle/input/stack-overflow-tag-network/stack_network_nodes.csv'

df=pd.read_csv(filename,encoding='ISO-8859-1')

df
filename1='/kaggle/input/stack-overflow-tag-network/stack_network_links.csv'

df1 = pd.read_csv(filename1, encoding='ISO-8859-2')

df1
G = nx.Graph()

for index, row in df.iterrows():

    G.add_node(row["name"],group = row["group"], nodesize = row["nodesize"] )

for index, row in df1.iterrows():

    G.add_edge(row["source"], row["target"], weight = row["value"])
print(nx.info(G))
def draw_graph(G,size):

    nodes = G.nodes()

    color_map = {1:'#f09494', 2:'#eebcbc', 3:'#72bbd0', 4:'#91f0a1', 5:'#629fff', 6:'#bcc2f2',7:'#eebcbc', 8:'#f1f0c0', 9:'#d2ffe7', 10:'#caf3a6', 11:'#ffdf55', 12:'#ef77aa',13:'#d6dcff', 14:'#d2f5f0'}

    node_color= [color_map[d['group']] for n,d in G.nodes(data=True)]

    node_size = [d['nodesize']*10 for n,d in G.nodes(data=True)]

    pos = nx.drawing.spring_layout(G,k=0.70,iterations=60)

    plt.figure(figsize=size)

    nx.draw_networkx(G,pos=pos,node_color=node_color,node_size=node_size,edge_color='#FFDEA2',edge_width=1)

    plt.show()
draw_graph(G,size=(30,30))