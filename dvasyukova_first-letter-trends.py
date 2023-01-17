%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.cm as cm
df = pd.read_csv('../input/NationalNames.csv')

df['FirstLetter'] = df.Name.str[0]

first = df.groupby(['Year','FirstLetter'])['Id'].count().unstack().fillna(0)

first.head()
norm = 100*first.div(first.sum(axis=1),axis=0)

normc = norm.cumsum(axis=1)

normc.head()
def lpos(heights):

    pos = np.zeros(heights.shape[0])

    pos[1:] += heights[:-1]

    pos = 0.5*(pos+heights)

    return pos
Nletters = normc.shape[1]

colors = cm.Pastel1(np.linspace(0.1, 1., Nletters))

x = normc.index

leftpos = pd.Series(lpos(normc.iloc[0,:].values),index=normc.columns)

rightpos = pd.Series(lpos(normc.iloc[-1,:].values),index=normc.columns)

fig, ax = plt.subplots(figsize=(12,12))

for (i,l) in enumerate(normc.columns[::-1]):

    ax.bar(x,normc[l],width=1, color=colors[i])

    if norm[l].iloc[0] > 1:

        ax.text(x[0]-3, leftpos[l], l, horizontalalignment='center',

                verticalalignment='center', backgroundcolor=colors[i])

    if norm[l].iloc[-1] > 1:

        ax.text(x[-1]+3, rightpos[l], l, horizontalalignment='center',

                verticalalignment='center', backgroundcolor=colors[i])

ax.set_xlim(x.min()-5, x.max()+5)

ax.set_ylim(0,100);

ax.set_title("% of names starting with letter",fontsize='x-large');
fig, ax = plt.subplots()

ax.plot(x, norm['K'], x, norm['C'])

ax.legend(['K','C'],loc=2)

ax.set_title('K vs C');
ax = (norm.iloc[-1,:] - norm.iloc[0,:]).plot.bar(rot=0)

ax.set_title('Changes from {} to {}'.format(x.min(),x.max()));