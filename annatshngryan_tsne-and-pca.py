# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_stand = scaler.transform(X)
from sklearn.cluster import KMeans, MiniBatchKMeans
mkmeans = MiniBatchKMeans(n_clusters=10)
mkmeans.fit(X_stand)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_stand)
tsne = TSNE(n_components=2, learning_rate=10)
X_tsne = tsne.fit_transform(X_stand)
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='rainbow')
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='rainbow')
