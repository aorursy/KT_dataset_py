import numpy as np

import pandas as pd

import csv

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
embeddings = pd.read_table('/kaggle/input/glove6b300dtxt/glove.6B.300d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
embeddings.head(10)
tsne = TSNE(n_components=2, random_state=0)
words = ['Riga', 'Country', 'Latvia', 'Capital']

colors = ['red', 'black', 'red', 'black']

v300d = [embeddings.loc[w.lower()] for w in words]
np.array(v300d).shape
v2d = tsne.fit_transform(v300d)

v2d.shape
plt.scatter(v2d[:, 0], v2d[:, 1], s=300, c=colors)



for c, label, x, y in zip(colors, words, v2d[:, 0], v2d[:, 1]):

    plt.annotate(label, xy=(x, y), xytext=(6, 6), textcoords="offset points", fontsize=42, color=c, fontweight='bold')



plt.xlim(-100,400)

plt.ylim(-400,100)

plt.grid()

plt.gcf().set_size_inches((30, 20))    

plt.show()