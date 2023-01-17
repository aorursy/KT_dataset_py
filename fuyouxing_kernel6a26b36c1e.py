# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import DBSCAN 

from itertools import cycle

from sklearn import metrics

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load data

data = np.loadtxt("/kaggle/input/data_perf.txt",delimiter = ",")

print(data[:5])
# Find the best epsilon

eps_grids       = np.r_[0.3 : 1.2 : 10j]

best_model      = None

best_label      = None

silh_scores     = []

silh_scores_max = 0

eps_best        = eps_grids[0]

for eps in eps_grids:

    # Train DBSCAN clustering model

    model = DBSCAN(eps=eps, min_samples=5)

    model.fit(data)

    # Extract labels

    labels = model.labels_

    # Extract performance metric 

    scores = metrics.silhouette_score(data, labels, sample_size = data.__len__())

    silh_scores.append(scores)

    print("eps --> {0:.1f}, silhouette score --> {1:.3f}".format(eps,scores))

    if scores > silh_scores_max:

        silh_scores_max = np.round(scores, 3)

        best_model, best_label = model, labels

        eps_best = eps

# Plot silhouette scores vs epsilon

plt.figure()

plt.bar(eps_grids, silh_scores, color = "darkgreen", align = "center", width = 0.08)

plt.title('Silhouette score vs epsilon')

plt.show()
# Check for unassigned datapoints in the labels

off = 1 if -1 in best_label else 0

print(off)

# Number of clusters in the data 

num_class = np.unique(best_label).__len__() - off 

num_class
# Extracts the core samples from the trained model

mark = np.zeros(data.shape, dtype=bool)

mark[best_model.core_sample_indices_] = True
# Plot resultant clusters 

labels_unique = set(best_label)

markers = cycle("vo^<>")

for labs, marker in zip(labels_unique, markers):

    # Use black dots for unassigned datapoints

    if labs == -1:

        marker = "."

    # Create mask for the current label

    idx = (best_label == labs)

    dts = data[idx & mark[:, 0]]

    plt.scatter(dts[:, 0], dts[:, 1], marker = marker, s = 80, facecolor = "k")



    dts = data[idx & ~mark[:, 0]]

    plt.scatter(dts[:, 0], dts[:, 1], marker=marker, s = 20, facecolor = "r")

plt.title('Data separated into clusters')

plt.show()