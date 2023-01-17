import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from functools import partial

from collections import defaultdict





# Load in the top1m dataset

df = pd.read_csv('../input/top-1m.csv', header=None, names=['Rank', 'Domain'])



# Get just the name (drop the .com, .net, etc.)

df['Name'] = df['Domain'].str.extract('((?:\w{2-3}\.)?\w+)')
# counting the times we see a transition from letter X to letter Y

occ = defaultdict(partial(defaultdict, int))



for name in df['Name']:

    for cur, next in zip(name, name[1:]):

        occ[cur][next] += 1



d = pd.DataFrame(occ).fillna(0)
# Compute the correlation matrix

corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap='RdBu', 

            vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap="viridis",

            vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()