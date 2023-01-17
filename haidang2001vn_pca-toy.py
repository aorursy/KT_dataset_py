import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns; sns.set()
rng = np.random.RandomState(1)

# print(rng.rand(2, 2))

# print(rng.randn(2, 200))

N = 10

X = np.dot(rng.rand(2, 2), rng.randn(2, N)).T  # X is centered

# X = rng.randn(2, N).T

print(X.shape)



# fig = plt.figure().add_subplot(111, projection='3d')



print(np.mean(X,0))

# fig.scatter(X[:, 0], X[:, 1], X[:, 2])

plt.scatter(X[:, 0], X[:, 1])

plt.axis('equal');



# Y  = rng.randn(2, 200).T

# plt.scatter(Y[:, 0], Y[:, 1])

# plt.axis('equal');
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X)
print(pca.components_)
print("explained_variance_", pca.explained_variance_) # The amount of variance explained by each of the selected components.

print("ratio", pca.explained_variance_ratio_)

print("singular_values_", pca.singular_values_)



# import numpy as np

# # centered_matrix = X - X.mean(axis=1)[:, np.newaxis]

# cov = np.dot(X.T,X)

# eigvals, eigvecs = np.linalg.eig(cov)

# print("eigvalue:", eigvals)

# print("eigvecs", eigvecs)

# print(np.sqrt(eigvals))
def draw_vector(v0, v1, ax=None):

    ax = ax or plt.gca()

    arrowprops=dict(arrowstyle='->',

                    linewidth=2,

                    shrinkA=0, shrinkB=0)

    ax.annotate('', v1, v0, arrowprops=arrowprops)



# plot data

plt.scatter(X[:, 0], X[:, 1], alpha=0.2)

for length, vector in zip(pca.explained_variance_, pca.components_):

    v = vector * 3 * np.sqrt(length)

    draw_vector(pca.mean_, pca.mean_ + v)

plt.axis('equal');
### try   explained_variance_ratio_  [0.97634101 0.02365899]

# pca = PCA(0.97)

# pca = PCA(0.98)



pca = PCA(n_components=1) 

pca.fit(X)

X_pca = pca.transform(X)

print("original shape:   ", X.shape)

print("transformed shape:", X_pca.shape)

print(pca.components_)

print(pca.explained_variance_)

print(X_pca)
X_new = pca.inverse_transform(X_pca)  # Transform data back to its original space.

plt.scatter(X[:, 0], X[:, 1], alpha=0.2,label='original')

plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8,label='projection')

plt.axis('equal');

plt.legend()