import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt

import numpy as np

import operator

%matplotlib inline
directory = '../input/'
responses = None

for i, chunk in enumerate(pd.read_csv(directory+'train_numeric.csv', chunksize = 10000, usecols=['Id','Response'])):

    if(responses is None):

        responses = chunk.copy()

    else:

        responses = pd.concat([responses, chunk.copy()])

    if i == 2:

        break
dates = None

for i, chunk in enumerate(pd.read_csv(directory+'train_date.csv', chunksize = 10000)):

    if(dates is None):

        dates = chunk.copy()

    else:

        dates = pd.concat([dates, chunk.copy()])

    if i == 2:

        break
alldata = pd.merge(dates, responses, on="Id")
alldata[alldata.columns[1:-1]] = alldata[alldata.columns[1:-1]].astype(float)

alldata[alldata.columns[1:-1]] = alldata[alldata.columns[1:-1]]-alldata[alldata.columns[1:-1]].min()

alldata[alldata.columns[1:-1]] = alldata[alldata.columns[1:-1]]/alldata[alldata.columns[1:-1]].max()

alldata.fillna(-1,inplace=1,axis=1)

remove = []

cols = alldata.columns[1:-1]

for i in range(len(cols)-1):

    v = alldata[cols[i]].values

    for j in range(i+1,len(cols)):

        if np.array_equal(v,alldata[cols[j]].values):

            remove.append(cols[j]),

alldata.drop(remove, axis=1, inplace=True),

remove = []

for col in alldata.columns[1:-1]:

    if alldata[col].std() == 0.0:

        remove.append(col)

alldata.drop(remove, axis=1, inplace=True)

pdt = alldata[alldata.Response==1]

ndt = alldata[alldata.Response==0]
posnet = nx.DiGraph()

for row in range(100):

    d = pdt.iloc[row][pdt.columns[1:-1]].to_dict()

    d = { k:v for k, v in d.items() if v!=-1.0 }

    sorted_x = sorted(d.items(), key=operator.itemgetter(1))

    prevk = None

    prevv = None

    for i, (k,v) in enumerate(sorted_x):

        if(prevk == None):

            prevk = k

            prevv = v

        else:

            posnet.add_weighted_edges_from([(prevk,k,v-prevv)])

            prevk = k

            prevv = v



plt.figure(figsize=(10,10))

nx.draw(posnet, with_labels=True)
negnet = nx.DiGraph()

for row in range(100):

    d = ndt.iloc[row][ndt.columns[1:-1]].to_dict()

    d = { k:v for k, v in d.items() if v!=-1.0 }

    sorted_x = sorted(d.items(), key=operator.itemgetter(1))

    prevk = None

    prevv = None

    for i, (k,v) in enumerate(sorted_x):

        if(prevk == None):

            prevk = k

            prevv = v

        else:

            negnet.add_weighted_edges_from([(prevk,k,v-prevv)])

            prevk = k

            prevv = v



plt.figure(figsize=(10,10))

nx.draw(negnet, with_labels=True)
def GetTopXNode(dct, topX=10):

    sorted_x = sorted(dct.items(), key=operator.itemgetter(1)) 

    top10 = [node[0] for node in sorted_x[-topX:]]

    return top10
plt.figure(figsize=(10,10))

top10connections = GetTopXNode(nx.degree(posnet))

print(top10connections)

sub = posnet.subgraph(top10connections)

nx.draw(sub, with_labels=True, font_size=20)
plt.figure(figsize=(10,10))

top10connections = GetTopXNode(nx.degree(negnet))

print(top10connections)

sub = negnet.subgraph(top10connections)

nx.draw(sub, with_labels=True, font_size=20)