# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/f_neighbors.csv')
edg = pd.concat([df['Words'],df['f_neighbor']],axis=1)
edg.at[172818, 'f_neighbor'] = str({'zyzzyvas'})
edg.at[172819, 'f_neighbor'] = 'set()'
edg['f_neighbor'] = edg.f_neighbor.apply(lambda x: {} if x == 'set()' or type(x) != 'string' else set(map(str.strip,str(x).replace("'","")[1:-1].split(','))))
# type(edg.at[0, 'f_neighbor'])
dic = df.set_index('Words')['f_neighbor'].to_dict()
dic['zyzzyva'] = str({'zyzzyvas'})
dic['zyzzyvas'] = 'set()'
# eval(dic['aa'])
for d in dic:
    if dic[d] == 'set()':
        dic[d] = set()
    else:
        dic[d] = eval(str(dic[d]))
# type(dic['aa'])
# dic
 # single edge (0,1)
G = nx.from_dict_of_lists(dic,create_using=nx.DiGraph())
G.number_of_nodes()
# G.number_of_edges()
# plt.draw()
G.number_of_edges()
# def allneighbors(df, key,depth = float('Inf')):
#     nn = [key]
#     ln = [key]
#     d = 0
#     e = df.edges()
#     if key not in df:
#         raise Exception("Key: \'{}\' is not in network graph".format(key))
#     while len(ln) > 0 and d < depth:
#         for lk in ln:
#             for n in df[lk]:
#                 nn.append(n)
#                 ln.append(n)
#             ln.remove(lk)
#         d += 1         
#     return {'nodes': set(nn),'depth': d}
def allneighbors(df, key,depth = float('Inf'),reverse = False):
    if reverse == True:
        df = nx.DiGraph.reverse(df)
    nn = [key]
    ln = [key]
    d = 0
    e = df.edges()
    for ke in df.edges(key):
        df.edges[ke]['depth'] = d
    if key not in df:
        raise Exception("Key: \'{}\' is not in network graph".format(key))
    while len(ln) > 0 and d < depth:
        for lk in ln:
            for n in df[lk]:
                nn.append(n)
                ln.append(n)
                for se in df.edges(key):
                    df.edges[se]['depth'] = d
            ln.remove(lk)
        d += 1
    return nx.Graph(df.subgraph(nn),Depth=d)
sk = 'bad'
sg = allneighbors(G,sk)
print('nodes : {}  edges : {}'.format(len(sg.nodes),len(sg.edges)))
# sg.edges['fried', 'friedcake']
# nx.dag_longest_path(G)
# nx.networkx.is_connected(G)

# import random as ran
# mycm = pd.Series([ran.randint(0,100)/100 for i in sg.nodes()],index=sg.nodes())
background = '#F5F5DC' #F5F5DC for beige, #F5F5F5 for light grey
prime = '#ADD8E6' #baby blue

options = {
    'with_labels': 'true',
    'node_size': 60,
    'node_shape': 'o',
#     'node_color': mycm,
#     'cmap': plt.cm.Blues,
    'font_size': 4,
    'edge_color':'#778899',
#     'edge_cmap' :,
    'alpha': 1,
    'width': 0.5,
    'font_color': '#232B2B', #Charleston Green
    'font_weight': 'light',
    'font_family': 'monospace',
    'style': 'dotted',
    'arrowstyle': '->',
    'arrowsize': 8,
    'arrows':False
}
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['text.color'] = '#1A1110' #Licorice
plt.rcParams['axes.titleweight'] = 'light'
plt.rcParams['font.size'] = 11
def draw_neighbors(df, sk, depth = float('Inf'), reverse = False):
    """
    draw a graph
    
    """
    
    sg = allneighbors(df, sk, depth, reverse)
    print('nodes: {}\nedges: {}\ndepth: {}'.format(len(sg.nodes),len(sg.edges),sg.graph['Depth']))
    mycm = pd.Series([prime if i == sk else background for i in sg.nodes()],index=sg.nodes())
    
    fig = plt.figure(num=1, dpi=300)
    nx.draw_spring(sg,node_color=mycm,k=1,threshold=1e-100,iterations=5000,**options) #(G, dim=2, k=None, pos=None, fixed=None, iterations=50, weight='weight', scale=1.0, center=None)
    fig.set_facecolor(background)
    plt.title("'{}' hooks".format(sk))
    plt.suptitle('depth: {}'.format((sg.graph['Depth'])),x=1, y=0, horizontalalignment='right', verticalalignment='bottom',fontsize=10)
    plt.legend(('words : {}'.format(len(sg.nodes)), 'hooks : {}'.format(len(sg.edges))),markerscale=.75, loc=(.9, 1), handlelength=1, fontsize=6)
    plt.show()
draw_neighbors(G, 'nonrepresentationalisms', float('Inf'), True)
nx.dag_longest_path(G)

import inspect
help(allneighbors)
dir(inspect.getsource)
print(inspect.getsource(inspect.unwrap))
# figure(num=1, figsize=20,20, dpi=600, facecolor=None, edgecolor=None, frameon=True, FigureClass=<class 'matplotlib.figure.Figure'>, clear=False)
help(plt.legend)
# nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
#     nx.draw_random(G, **options)
#     plt.show()
#     nx.draw_spring(G, **options)
#     plt.show()
# draw_networkx_nodes(G, pos, nodelist=None, node_size=300, node_color='r', node_shape='o', alpha=1.0, cmap=None, vmin=None, vmax=None, ax=None, linewidths=None, edgecolors=None, label=None, **kwds)
# nx.draw(G, with_labels=True, node_size=0, style='dotted', edge_color='grey',pos=nx.spring_layout(G),nodelist=[],edgelist=[])
# G=nx.from_pandas_dataframe(df, 'Words', 'f_neighbor')