# Import our relevant libraries

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/data.csv')

data.head()
# Drop the id column

data = data.drop('id', axis=1)

# Convert the diagnosis column to numeric format

data['diagnosis'] = data['diagnosis'].factorize()[0]

# Fill all Null values with zero

data = data.fillna(value=0)

# Store the diagnosis column in a target object and then drop it

target = data['diagnosis']

data = data.drop('diagnosis', axis=1)
from sklearn.decomposition import PCA # Principal Component Analysis module

from sklearn.manifold import TSNE # TSNE module
# Turn dataframe into arrays

X = data.values



# Invoke the PCA method. Since this is a binary classification problem

# let's call n_components = 2

pca = PCA(n_components=2)

pca_2d = pca.fit_transform(X)



# Invoke the TSNE method

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)

tsne_results = tsne.fit_transform(X)
# Plot the TSNE and PCA visuals side-by-side

plt.figure(figsize = (16,11))

plt.subplot(121)

plt.scatter(pca_2d[:,0],pca_2d[:,1], c = target, 

            cmap = "coolwarm", edgecolor = "None", alpha=0.35)

plt.colorbar()

plt.title('PCA Scatter Plot')

plt.subplot(122)

plt.scatter(tsne_results[:,0],tsne_results[:,1],  c = target, 

            cmap = "coolwarm", edgecolor = "None", alpha=0.35)

plt.colorbar()

plt.title('TSNE Scatter Plot')

plt.show()
# Calling Sklearn scaling method

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
# Invoke the PCA method on the standardised data

pca = PCA(n_components=2)

pca_2d_std = pca.fit_transform(X_std)



# Invoke the TSNE method

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)

tsne_results_std = tsne.fit_transform(X_std)
# Plot the TSNE and PCA visuals side-by-side

plt.figure(figsize = (16,11))

plt.subplot(121)

plt.scatter(pca_2d_std[:,0],pca_2d_std[:,1], c = target, 

            cmap = "RdYlGn", edgecolor = "None", alpha=0.35)

plt.colorbar()

plt.title('PCA Scatter Plot')

plt.subplot(122)

plt.scatter(tsne_results_std[:,0],tsne_results_std[:,1],  c = target, 

            cmap = "RdYlGn", edgecolor = "None", alpha=0.35)

plt.colorbar()

plt.title('TSNE Scatter Plot')

plt.show()