# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.spatial.distance import squareform, pdist



import matplotlib.pyplot as plt





mat = np.array([[0.0, 1.0, 4.0, 5.0], [1.0, 0.0, 2.0, 6.0], [4.0, 2.0, 0.0, 3.0], [5.0, 6.0, 3.0, 0.0]])

dists = squareform(mat)

linkage_matrix = linkage(dists, "complete")

dendrogram(linkage_matrix, labels=["A", "B", "C", "D"])

plt.title("Complete Link - 1")

plt.show()