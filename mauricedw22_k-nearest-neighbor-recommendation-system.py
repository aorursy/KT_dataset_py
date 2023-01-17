# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import NearestNeighbors

# Sample dataset containing with 6 rows, 2 features each.

X = np.array([[-11, 20, 30], [43, -45, 70], [-45, 1, 18], [-9, 33, 21], [-4, 20, 25], [-7, 40, 30]])
# Here we instantiate a nearest neighbor object to fit onto dataset X.

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)



# Next we find k nearest neighbor for each point in object X.

distances, indices = nbrs.kneighbors(X)
# Checking print-out of the indices of neighbors for each record in object X.

print(indices)
# New data point check against our k neighbors classifier.

# To search object X and identify the closest related record, we call the kneighbors() function on the new data point

print(nbrs.kneighbors([[-2, 4, 6]])) 