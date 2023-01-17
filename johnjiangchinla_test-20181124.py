# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Y-STR.csv', header=0, index_col=0)
H_num = data["Numble"]
H_index = data.index
t_data = data.drop(columns=["Numble"])
mis = np.full((9,9), np.nan)
dis = np.full((9,9), np.nan)                           # 变化矩阵
for i in range(9):
    data_lm = t_data.diff(i+1).dropna(how="all")
    p_list = (data_lm !=0 ).astype(int).sum(axis=1)     # 计算改变列数
    i_list = data_lm.apply(abs).T.sum()                 # 计算改变总数
    
    for j in range(len(p_list)):
        # dis_map[i+j+1, j] = [i_list[j], p_list[j]]
        mis[i+j+1, j] = i_list[j]
        mis[j, i+j+1] = i_list[j]
        dis[i+j+1, j] = p_list[j]
        dis[j, i+j+1] = p_list[j]
print(mis)
print(dis)
def max(i):
    return i if i==1 else np.inf
mis_v = np.vectorize(max)(mis)
print(mis_v)
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
for index in H_index:
    G.add_node(index, nodesize=H_num[index])
# for edge in edges:
#     G.add_edges_from([('H' + str(edge[0]+1), 'H' +str(edge[1]+1))])
for i in range(len(mis_v)):
    for j in range(len(mis_v)):
        if mis_v[i, j] == 1:
            G.add_edge('H' + str(i+1), 'H' +str(j+1))
sizes = H_num*300
nx.draw(G, node_size= sizes, with_labels=True, pos=nx.spring_layout(G, k=0.25, iterations=50))
ax = plt.gca()
ax.collections[0].set_edgecolor("#555555") 
plt.show()