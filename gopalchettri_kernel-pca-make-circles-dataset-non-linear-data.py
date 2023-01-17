import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.datasets import make_circles

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.decomposition import KernelPCA
# Generate a dataset where the variations cannot be captured by a straight line

np.random.seed(0)

x,y = make_circles(n_samples=400, factor=.2, noise=0.02)

print('x =', x)
print('y =', y)
# Plot the generated dataset

plt.close('all')

plt.figure(1)

plt.title("Original Space")

plt.scatter(x[:, 0], x[:,1],c =y)

plt.xlabel("$x_1$")

plt.ylabel("$x_2$")

plt.show()
# Try to fit the data using normal PCA

pca = PCA(n_components=2)

pca.fit(x)

x_pca = pca.transform(x)
# Summarize the compnents

print("Shape = ",x_pca.shape)

print("The estimated number of components = ",pca.components_)

print("Explained Variance = ", pca.explained_variance_ratio_ )
# Plot the first two principal components of this dataset.

# We will plot the dataset using only the first principal component.

plt.figure(2)

plt.title("PCA")

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)

plt.xlabel("$x_1$")

plt.ylabel("$x_2$")

plt.show()
# Plot using the first component from normal pca

class_1_indx = np.where(y==0)[0]

class_2_indx = np.where(y==1)[0]



plt.figure(3)

plt.title("PCA - One Component")

plt.scatter(x_pca[class_1_indx, 0], np.zeros(len(class_1_indx)), color='red')

plt.scatter(x_pca[class_2_indx, 0], np.zeros(len(class_2_indx)), color='blue')

plt.show()
# Create Kernel PCA object in scikit learn, specifying a type of kernel as parameter

kpca = KernelPCA(kernel="rbf", gamma=10)

# perform kernelPCA

kpca.fit(x)

x_kpca = kpca.transform(x)

print(x_kpca)
# Summarize the compnents

print("Shape = ",x_kpca.shape)
# plot the first two components

plt.figure(4)

plt.title("Kernel PCA")

plt.scatter(x_kpca[:, 0], x_kpca[:, 1], c=y)

plt.xlabel("$x_1$")

plt.ylabel("$x_2$")

plt.show()
from sklearn.datasets import make_moons

x,y = make_moons(100)

plt.figure(5)

plt.title("Non Linear Data")

plt.scatter(x[:,0],x[:,1],c=y)

plt.xlabel("$x_1$")

plt.ylabel("$x_2$")

plt.savefig('fig-7.png')

plt.show()
# Singular Value Decomposition (SVD) is yet another matrix decomposition technique that

# can be used to tackle the curse of the dimensionality problem. It can be used to find the best

# approximation of the original data using fewer dimensions. Unlike PCA, SVD works on the

# original data matrix.
from scipy.linalg import svd

from sklearn.datasets import load_iris



# Load Iris dataset

data = load_iris()

x_ = data['data']

y_ = data['target']
print(x_.shape)
# Proceed by scaling the x variable w.r.t its mean,

from sklearn.preprocessing import scale

x_s = scale(x_,with_mean=True,with_std=False,axis=0)
# Decompose the matrix using SVD technique.We will use SVD

# implementation in scipy.

from scipy.linalg import svd

U,S,V = svd(x_s,full_matrices=False)
# Approximate the original matrix by selecting only the first two

# singular values.

x_t = U[:,:2]
print(x_t.shape)
# Finally we plot the datasets with the reduced components.

plt.figure(5)

plt.scatter(x_t[:,0],x_t[:,1],c=y_)

plt.xlabel("Component 1")

plt.ylabel("Component 2")

plt.show()