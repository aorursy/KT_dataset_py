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
import networkx as nx

import community

import matplotlib.pyplot as plt
train_df = pd.read_csv('../input/data_set_ALL_AML_train.csv')

train_df.head()
train_df = train_df.drop(['Gene Description','Gene Accession Number'],axis=1)

call_cols = ['call']+['call.'+str(i) for i in range(1,38)]

train_df = train_df.drop(call_cols,axis=1)

train_df.columns
actual_df = pd.read_csv('../input/actual.csv')

actual_df.head()
train_df = train_df.T

train_df.head()
train_df['cancer'] = list(actual_df['cancer'])[:38]

train_df.shape
train_df_ALL = train_df[[col for col in train_df.columns if col!='cancer']][train_df['cancer']=='ALL']
train_df_AML = train_df[[col for col in train_df.columns if col!='cancer']][train_df['cancer']=='AML']
train_ALL = train_df_ALL.as_matrix().astype('int')

from sklearn.preprocessing import MinMaxScaler as scaler

min_scaler = scaler()

train_ALL_norm = min_scaler.fit_transform(train_ALL)

corr_ALL = np.corrcoef(train_ALL_norm.T)

corr_ALL
corr_ALL[corr_ALL<0.5] = 0

G = nx.karate_club_graph()

nx.draw_spring(G, with_labels=True)
G_ALL = nx.convert_matrix.from_numpy_matrix(corr_ALL)

nx.draw_spring(G_ALL, with_labels=True)
# Modularity between a particular node in pi and u in the Graph G



def q(u,pi,G):

    # find for the node 'u'

    s = pi[u] # s stores the uth index node (as pi[u] = u)

    s_nodes = [k for k,v in pi.items() if v == s] # check all the nodes connected with given node index s

    G1 = G.subgraph(s_nodes) # create a subgraph of them

    # average degree of node u in the subgraph  with the given selected nodes

    i_u = G1.degree(u)/2

    d_u = G.degree(u)

    # sum of the degree of the nodes of the graph

    D_s = sum(list((dict(G.degree(s_nodes)).values())))

    m = sum(list((dict(G.degree()).values())))

    return i_u - ((d_u*D_s)/m)
# Total modularity of the graph with respect to Graph G

def Q(pi,G):

    Q_total = 0

    for u in G.nodes():

        Q_total = Q_total + q(u,pi,G)

    m = sum(list((dict(G.degree()).values())))

    return Q_total/m
# Switch for community for a given node 'u' in the Graph G

def comm_switch_ind(u,G,pi, epsilon):

    V = list(G[u]) #list of all the nodes in G connected to u

    V_comm = list(set([pi[v] for v in V])) # find those nodes common with pi

    prev_pi = pi.copy()

    prev_mod = q(u,pi,G)  ### check previous modularity

    mod = prev_mod

    for comm in V_comm:

        pi[u] = comm # update with the common nodes and if merged check the new modularity 

        new_mod = q(u,pi,G) 

        # if the new modularity is above the given modularity by a slight amount in epsilon

        if abs(new_mod - mod) > epsilon:

            mod = new_mod

            mod_comm = comm # u[date the new community

    if abs(mod - prev_mod) > epsilon:

        prev_pi[u] = mod_comm

        return prev_pi

    else:

        return prev_pi
def merge_comm(pi,G, epsilon):

    comm_list = list(set(pi.values())) # find the distinct elements of the communities

    prev_pi = pi.copy()

    prev_mod = Q(pi,G)   

    mod = prev_mod

    for s1 in comm_list:

        for s2 in comm_list:

            if s1!=s2 : # if unequal communities

                pi = prev_pi.copy()

                s1_nodes = [k for k,v in pi.items() if v == s1]

                for u in s1_nodes:

                    pi[u] = s2

                new_mod = Q(pi,G)    ### here

                if abs(new_mod - mod) > epsilon:

                    mod = new_mod

                    prev_pi = pi.copy()

    

    return prev_pi, mod


def CDG(G,nIter,epsilon):

    merge_plot = []

    pi = {v:v for v in list(G.nodes())}

    prev_pi = pi

    t = 0

    modul = 0

    while(True):

        if t == nIter:

            break

        for u in G.nodes():

            pi =  comm_switch_ind(u,G,pi,epsilon)

        pi, modul = merge_comm(pi,G,epsilon)

        merge_plot.append(modul)

        

        t = t+1

    """"t = 0

    while(True):

        if t == 10:

            break

        pi = merge_comm(pi,G)

        print(pi)

        t = t+1"""

    return pi, (sum(merge_plot)/nIter*1.0)
nIter = 10

mer = []

epsilon = -0.00001

while(nIter < 30):

    pi_opt , avg = CDG(G,nIter,epsilon)

    mer.append(avg)

    nIter += 1
import seaborn as sns
pi = {v:v for v in list(G.nodes())}
s= pi[1]

s_nodes = [k for k,v in pi.items() if v == s]
ax = sns.barplot([i for i in range(len(mer))],mer)

ax.set(xlabel='No of Iterations', ylabel='Merge')

comms = list(set(pi_opt.values()))

comms
values = [pi_opt.get(node)*1000 for node in G.nodes()]

nx.draw_spring(G, node_color = values, node_size=200, with_labels=True)
from networkx .generators.classic import barbell_graph

G_barbel = barbell_graph(10,5)

nx.draw_spring(G_barbel,with_labels=True)
dict(G.degree()).values()