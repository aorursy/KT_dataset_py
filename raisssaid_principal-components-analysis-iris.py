import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
# Loading dataset...
iris = load_iris()
# Dimensions matrix
X = iris.data
# Labels
y = iris.target
target_names = iris.target_names

# Principal Components Analysis
# Using PCA to switch from 4 dim to 2 dim
pca = PCA(n_components = 2)
# Transform X to a new space (2-dim)
X_r = pca.fit(X).transform(X)
# Shape of X_r
X_r.shape
# Percentage of variance explained by each of the selected components.
print(
    "Percentage of variance explained by each of the selected components : {}".format(
        pca.explained_variance_ratio_
        )
    )
colors = ['red', 'blue', 'green']

plt.figure()
for color, i, target_name, in zip(colors, [0, 1, 2], target_names):
  plt.scatter(
      X_r[y == i, 0], X_r[y == i, 1],
      color = color,
      alpha = 0.8, lw = 2,
      label = target_name
      )
plt.legend(loc = 'best', shadow = False, scatterpoints = 1)
plt.title('Iris en 2D')