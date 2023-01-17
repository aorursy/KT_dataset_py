import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

from learntools.core import binder; binder.bind(globals())
from learntools.embeddings.ex4_tsne import *

input_dir = '../input/visualizing-embeddings-with-t-sne'
csv_path = os.path.join(input_dir, 'movies_tsne.csv')
df = pd.read_csv(csv_path, index_col=0)
FS = (13, 9)
fig, ax = plt.subplots(figsize=FS)

c = np.random.rand(len(df))

pts = ax.scatter(df.x, df.y, c=c)

cbar = fig.colorbar(pts)
#part1.solution()
# Your code goes here
fig, ax = plt.subplots(figsize=FS)
c = df.mean_rating
pts = ax.scatter(df.x, df.y, c=c)
cbar = fig.colorbar(pts)
#part2.hint()
#part2.solution()
fig, ax = plt.subplots(figsize=FS)

c = df.n_ratings

pts = ax.scatter(df.x, df.y, c=c)

cbar = fig.colorbar(pts)
#part3.solution()