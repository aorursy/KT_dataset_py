import networkx as nx

import pydot

from hmmlearn import hmm

import pandas as pd

import numpy as np

import itertools

import matplotlib.pyplot as plt

import seaborn as sns

import pickle

%matplotlib inline



def markov_p(data):

  # from https://www.kaggle.com/friedchips/on-markov-chains-and-the-competition-data

  channel_range = np.unique(data)

  channel_bins = np.append(channel_range, 11)

  data_next = np.roll(data, 1)

  matrix = []

  for i in channel_range:

    current_row = np.histogram(data_next[data == i], bins=channel_bins)[0]

    current_row = current_row / np.sum(current_row)

    matrix.append(current_row)

  return np.array(matrix)



train_clean = pd.read_csv("../input/data-without-drift/train_clean.csv")

X, y = train_clean["signal"].values, train_clean["open_channels"].values



sequences = np.zeros((X.shape[0], 1), dtype=np.float32)

# max. 1 open channel, low probability

sequences[0 * 500000:1 * 500000] = 0

sequences[1 * 500000:2 * 500000] = 0

# max. 1 open channel, high probability

sequences[2 * 500000:3 * 500000] = 1

sequences[6 * 500000:7 * 500000] = 1

# max. 3 open channels

sequences[3 * 500000:4 * 500000] = 2

sequences[7 * 500000:8 * 500000] = 2

# max. 5 open channels

sequences[5 * 500000:6 * 500000] = 3

sequences[8 * 500000:9 * 500000] = 3

# max. 10 open channels

sequences[4 * 500000:5 * 500000] = 4

sequences[9 * 500000:10 * 500000] = 4



for sequence, components in [(0, 2), (1, 2), (2, 4), (3, 6), (4, 11)]:

  print(f"Sequence {sequence} with {components-1} open channel(s)")



  X_part = X[np.flatnonzero(sequences == sequence)]

  y_part = y[np.flatnonzero(sequences == sequence)]



  h = hmm.GaussianHMM(n_components=components, covariance_type="tied", verbose=False, n_iter=200, random_state=3)

  h.fit(X_part.reshape((-1, 1)))



  with open(f"{sequence}.pkl", "wb") as f:

    pickle.dump(h, f)



  diff = 0

  for a, b in zip(sorted(markov_p(y_part).flatten()), sorted(h.transmat_.flatten())):

        diff += np.abs(a - b)



  print(f"MAE between histogram and viterbi transition matrix: {diff}")

  print(f"Score: {h.score(X_part.reshape(-1, 1))}")

  print(f"Acc: {np.sum((h.predict(X_part.reshape((-1, 1))) == y_part))/X_part.shape[0]}")



  fig, axes = plt.subplots(1, 1)

  fig.set_size_inches(10, 5)

  fig.subplots_adjust(wspace=0.05, hspace=0.05)

    

  sns.heatmap(h.transmat_, annot=True, fmt='.3f', cmap='Blues', cbar=False, ax=axes, vmin=0, vmax=0.5, linewidths=2)

  plt.show()



  G = nx.MultiDiGraph()

  for i, j in itertools.permutations(range(components), 2):

    percentage = h.transmat_[i][j]

    if percentage == 0:

        continue



    G.add_edge(i, j, weight=percentage, label="{:.03f}".format(percentage))



  nx.drawing.nx_pydot.write_dot(G, f"{sequence}.dot")

  (graph,) = pydot.graph_from_dot_file(f"{sequence}.dot")

  graph.write_png(f"{sequence}.png")



  plt.figure()

  img = plt.imread(f"{sequence}.png")

  plt.imshow(img)

  plt.show()