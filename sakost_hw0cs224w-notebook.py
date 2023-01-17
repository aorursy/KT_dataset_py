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
!python -m pip install snap-stanford
import snap
G1 = snap.LoadEdgeList(snap.PNGraph, '/kaggle/input/wiki-vote/wiki-Vote.txt', 0, 1)
G2 = snap.LoadEdgeList(snap.PNGraph, "/kaggle/input/hw0cs224w/stackoverflow-Java.txt", 0, 1)
G1.GetNodes()
self_looped = 0

for edge in G1.Edges():

    if edge.GetSrcNId() == edge.GetDstNId():

        self_looped += 1

self_looped
directed = 0

for edge in G1.Edges():

    if edge.GetSrcNId() != edge.GetDstNId():

        directed += 1

directed
undirected = 0

cache = set()

for edge in G1.Edges():

    from_, to = edge.GetSrcNId(), edge.GetDstNId()

    if (to, from_) in cache:

        undirected += 1

    else:

        cache.add((from_, to))

undirected
reciprocated = 0

cache = set()

for edge in G1.Edges():

    from_, to = edge.GetSrcNId(), edge.GetDstNId()

    if (to, from_) in cache and to != from_:

        reciprocated += 1

    else:

        cache.add((from_, to))

reciprocated
zod = 0

for node in G1.Nodes():

    if node.GetOutDeg() == 0:

        zod += 1

zod
zid = 0

for node in G1.Nodes():

    if node.GetInDeg() == 0:

        zid += 1

zid
mt10 = 0

for node in G1.Nodes():

    if node.GetOutDeg() > 10:

        mt10 += 1

mt10
lt10 = 0

for node in G1.Nodes():

    if node.GetInDeg() < 10:

        lt10 += 1

lt10
from collections import defaultdict



from matplotlib import pyplot as plt



count_out_degrees = defaultdict(int)



for node in G1.Nodes():

    count_out_degrees[node.GetOutDeg()] += 1





n_data = np.array(sorted(list(count_out_degrees.items())))



n_data = n_data[n_data[:, 0] != 0]



logx = np.log10(n_data[:, 0])

logy = np.log10(n_data[:, 1])

             

plt.scatter(logx, logy, s=7);
a, b = np.polyfit(logx, logy, 1)



plt.plot(logx, np.log10((10**logy - (a * 10**logx + b)) ** 2));
Components = snap.TCnComV()

snap.GetWccs(G2, Components)

Components.Len()
max_component = snap.GetMxWcc(G2)

edges = max_component.GetEdges()

nodes = max_component.GetNodes()



print('Nodes in max weakly connected component:', nodes)

print('And edges:', edges)
TOP = 3



PRankH = snap.TIntFltH()



snap.GetPageRank(G2, PRankH)



d = {k: PRankH[k] for k in PRankH}



for i in range(TOP):

    id_ = max(d, key=lambda x: d[x])

    print('id of ',i + 1, 'central node is', id_)

    del d[id_]
NIdHubH = snap.TIntFltH()

NIdAuthH = snap.TIntFltH()

snap.GetHits(G2, NIdHubH, NIdAuthH)



dHub = {k: NIdHubH[k] for k in NIdHubH}

dAuth = {k: NIdAuthH[k] for k in NIdAuthH}



for i in range(TOP):

    id_Hub = max(dHub, key=lambda x: dHub[x])

    id_Auth = max(dAuth, key=lambda x: dAuth[x])

    print('id of ', i + 1, 'hub node is', id_Hub)

    print('id of ', i + 1, 'auth node is', id_Auth)

    del dHub[id_Hub]

    del dAuth[id_Auth]