# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import scipy as sp

import seaborn as sns

import matplotlib.pyplot as plt

import csv

%matplotlib inline
# load the data

brain = np.genfromtxt('../input/day3_data.csv', delimiter=',')

rt = np.genfromtxt('../input/day3_rt.csv', delimiter=',')

cond = np.genfromtxt('../input/day3_condition.csv', delimiter=',')

with open(r'../input/day3_regions.csv') as f:

    region = list()

    for row in csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE):

        region.extend(row)
print('Data:', np.shape(brain), type(brain), sep = " ")

print('RT:',np.shape(rt),  type(rt), sep = " ")

print('cond:',np.shape(cond),  type(cond), sep = " ")

print('regions:',np.shape(region),  type(region), sep = " ")
plt.hist(rt)

plt.xlabel('RT (ms)')

plt.ylabel('Frequency')
kernel = sp.stats.gaussian_kde(rt)

pts = np.linspace(100,500,1000)



plt.hist(rt,density=True)

plt.plot(pts,kernel.evaluate(pts), color='black')

plt.xlabel('RT (ms)')

plt.ylabel('Frequency')
fig = plt.figure(figsize=(18,6))

plt.plot(rt)

plt.xlabel('Trial')

plt.ylabel('RT (ms)')
nTrial = len(rt)

trial = np.linspace(1,nTrial,nTrial)

[rho, p] = sp.stats.spearmanr(rt,trial)

print(rho)

print(p)
fig = plt.figure(figsize=(18,6))

plt.plot(rt)

plt.xlabel('Trial')

plt.ylabel('RT (ms)')

plt.title("".join(('Spearmans rho: ', str(round(rho,2)), ' p-value: ', str(p))))

plt.plot(np.unique(range(nTrial)), np.poly1d(np.polyfit(range(nTrial), rt, 1))(np.unique(range(nTrial))), color = "red",

        linewidth='3')
RT0 = rt[cond == 0]

RT1 = rt[cond == 1]

RT2 = rt[cond == 2]



sp.stats.f_oneway(RT0,RT1,RT2)
plt.boxplot([RT0,RT1,RT2],notch=True)

plt.xlabel('Condition')

plt.ylabel('RT (ms)')
import random

fig,axes = plt.subplots(2,3,figsize=(12,10))

axes = axes.ravel()



nSample = 6

elecs = random.sample(range(0, 45), nSample)

for i,e in enumerate(elecs):

    curr = brain[e,]

    axes[i].hist(curr)

    axes[i].title.set_text("".join(('Elec ', str(e))))

    axes[i].set_xlabel("Brain Activity")

    axes[i].set_ylabel("Frequency")
fig,axes = plt.subplots(2,3,figsize=(12,10))

axes = axes.ravel()



for i,e in enumerate(elecs):

    curr = brain[e,]

    sp.stats.probplot(curr,plot=axes[i])
fig,axes = plt.subplots(2,3,figsize=(12,10))

axes = axes.ravel()



for i,e in enumerate(elecs):

    curr = brain[e,]

    axes[i].hist(curr[cond==1],alpha=0.6,density=True)

    axes[i].hist(curr[cond==2],color='red',alpha=0.6,density=True)

    axes[i].title.set_text("".join(('Elec ', str(e))))

    axes[i].set_xlabel("Brain Activity")

    axes[i].set_ylabel("Frequency")
nElec = np.shape(brain)[0]



ps = list()

for i in range(nElec):

    curr = brain[i,]

    [u,p] = sp.stats.mannwhitneyu(curr[cond==1],curr[cond==2])

    ps.append(p)
sig_idx = [x < (0.05/nElec) for x in ps]

print(np.sum(sig_idx))



for i,x in enumerate(sig_idx):

    if x:

        print(region[i])
sig_elec = np.transpose(brain[sig_idx,:])

[u,p] = sp.stats.mannwhitneyu(sig_elec[cond==1],sig_elec[cond==2])

u
rs = list()

corr_ps = list()

for i in range(nElec):

    curr = brain[i,]

    [r,p] = sp.stats.pearsonr(curr,rt)

    ps.append(p)
sig_idx = [x < (0.05) for x in corr_ps]

print(np.sum(sig_idx))
def cohens(x,y):

    if (np.isnan(x).any() or np.isnan(y).any()):

        warnings.warn("One of your samples has nans in it.")

    m1 = np.mean(x)

    m2 = np.mean(y)

    std1 = np.std(x)**2

    std2 = np.std(y)**2

    d = (m2 - m1)/np.sqrt((std1 + std2)/2)

    return d
import warnings

x = np.random.normal(0, 1, 100)

y = np.random.normal(.2, 1, 100)

y[0] = np.nan

cohens(x,y)
x = np.random.normal(0, 1, 100)

y = np.random.normal(.2, 1, 100)



cohens(x,y)
cohens(x.tolist(),y.tolist())
eta = list()

for i in range(nElec):

    curr = brain[i,]

    d = cohens(curr[cond==1],curr[cond==2])

    eta.append(d)
plt.hist(eta,density=True,bins=20)
# initialize A

A = np.empty((nElec,nElec))

for i in range(nElec):

    for j in range(i,nElec):

        [curr,_] = sp.stats.pearsonr(brain[i,],brain[j,])

        A[i,j] = curr

        A[j,i] = curr

fig = plt.figure(figsize = (12,10))

sns.heatmap(A - np.eye(nElec),cmap='magma')
import networkx as nx
# Threshold:

A[A<0.1] = 0

sns.heatmap(A)
G=nx.from_numpy_matrix(A)

type(G)
pos = nx.kamada_kawai_layout(G)

colors = range(len(G.edges))

fig = plt.figure(figsize=(10,10))

nx.draw(G, pos, node_color='#A0CBE2', edge_color=colors,

        width=4, edge_cmap=plt.cm.Blues, with_labels=True)

plt.show()