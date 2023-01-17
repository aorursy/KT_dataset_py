import numpy as np

import pandas as pd

import  matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv')

dissimilar = train[train.is_duplicate==0]

similar = train[train.is_duplicate==1]
plt.scatter(similar.qid1[::100],similar.qid2[::100])
plt.scatter(dissimilar.qid1[::100],dissimilar.qid2[::100])
p1 = np.array([0.0,0.0]) # Origin

p2 = np.array([1.0, 1.0]) # Approximate to the line

ps = similar[['qid1','qid2']].values[::100]

dist = []

for i in range(ps.shape[0]):

    dist.append(np.sign(np.cross(p2-p1, p1-ps[i,:]))*np.linalg.norm(np.cross(p2-p1, p1-ps[i,:]))/np.linalg.norm(p2-p1))

plt.plot(np.array(dist))
_ = plt.hist(dist,bins=100)
p1 = np.array([0.0,0.0]) # Origin

p2 = np.array([1.0, 1.0]) # Approximate to the line

ps = dissimilar[['qid1','qid2']].values[::100]

dist = []

for i in range(ps.shape[0]):

    dist.append(np.sign(np.cross(p2-p1, p1-ps[i,:]))*np.linalg.norm(np.cross(p2-p1, p1-ps[i,:]))/np.linalg.norm(p2-p1))

plt.plot(np.array(dist))
_ = plt.hist(dist,bins=100)