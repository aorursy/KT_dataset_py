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
path = '/kaggle/input/ml-in-graphs-hw0/wiki-Vote.txt'
G = snap.LoadEdgeList(snap.PNGraph, path, 0, 1)
G.GetNodes()
snap.CntSelfEdges(G)
snap.CntUniqDirEdges(G)
snap.CntUniqUndirEdges(G)
snap.CntUniqBiDirEdges(G)
snap.CntOutDegNodes(G,0)
snap.CntInDegNodes(G,0)
#cnt <= 10
cntNodes = 0
for i in range(11):
    cntNodes += snap.CntOutDegNodes(G,i)
G.GetNodes() - cntNodes
#cnt <= 10
cntNodes = 0
for i in range(10):
    cntNodes += snap.CntInDegNodes(G,i)
cntNodes
import matplotlib.pyplot as plt
InDegV = snap.TIntPrV()
snap.GetOutDegCnt(G,InDegV)
pair = [[p.GetVal1(), p.GetVal2()] for p in InDegV if p.GetVal1() > 0]
x = [np.log(p[0]) for p in pair]
y = [np.log(p[1]) for p in pair]
plt.scatter(x,y)
import statsmodels.api as sm
lr = sm.OLS(y, sm.add_constant(x))
lr_res = lr.fit()
lr_res.summary()
lr_res.predict(x)
plt.scatter(x,y)
plt.plot(x,lr_res.predict(sm.add_constant(x)), color = 'r')
path = '/kaggle/input/ml-in-graphs-hw0/stackoverflow-Java.txt'
G2 = snap.LoadEdgeList(snap.PNGraph, path, 0, 1)
wcc = snap.TCnComV()
snap.GetWccs(G2, wcc)
len(wcc)
max_wcc = snap.GetMxWcc(G2)
print(max_wcc.GetEdges())
print(max_wcc.GetNodes())
PRankH = snap.TIntFltH()
snap.GetPageRank(G2,PRankH)
pair = [[k,PRankH[k]] for k in PRankH]
pair = sorted(pair, key = lambda x: x[1], reverse = True)
[p[0] for p in pair[:3]]
NIdHubH = snap.TIntFltH()
NIdAuthH = snap.TIntFltH()
snap.GetHits(G2, NIdHubH, NIdAuthH)
hub = [[k,NIdHubH[k]] for k in NIdHubH]
hub = sorted(hub, key = lambda x: x[1], reverse = True)
print([p[0] for p in hub[:3]])

aut = [[k,NIdAuthH[k]] for k in NIdAuthH]
aut = sorted(aut, key = lambda x: x[1], reverse = True)
print([p[0] for p in aut[:3]])