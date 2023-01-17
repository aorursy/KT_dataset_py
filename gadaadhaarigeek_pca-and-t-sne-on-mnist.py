# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

data.head()
# Original shape of data 

data.shape
# Just to make computations faster we will take a subset of images

# data = data.head(10000)
# Storing the labels 

label = data["label"]



# dropping the label column from the data as we don't need them for PCA

data = data.drop("label", axis=1)
# Shape of the data of pixels 

# Each image is of 28x28 pixels 

print(data.shape)
# Shape of label

print(label.shape)
# Plot one image 

plt.figure(figsize=(4, 4))

idx = 6

x = data.iloc[idx].as_matrix().reshape(28, 28)



for i in range(x.shape[0]):

    print()

    for j in range(x.shape[1]):

        print("{:3}".format(x[i][j]), end="")
# Above is looking like 7

# Now let's actually plot the number 

plt.figure(figsize=(4, 4))

idx = 6

x = data.iloc[idx].as_matrix().reshape(28, 28)



plt.imshow(x, interpolation = "none", cmap="gray")

plt.show()



print(label[idx])
print("Shape of sampled data: ", data.shape)
# Standardizing the data

# PCA requires the data to be scaled. (Why ?)

from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(data)

print(standardized_data.shape)
sampled_data = standardized_data



# Covariance matrix of shape 784x784 

# featureXfeature

covar_mat = np.matmul(sampled_data.T, sampled_data)

print("Shape of covariance matrix: ", covar_mat.shape)
from scipy.linalg import eigh



# eighvals takes the (low, high) as indices of eigen_values 

# eigen values are returned in ascending order

# We are just selecting highest two values of eigenvalues

eigen_values, eigen_vectors = eigh(a=covar_mat, eigvals=(782, 783))
print("Shape of eigen vectors: ", eigen_vectors.shape)
# Eigen values

eigen_values
eigen_vectors = eigen_vectors.T

print("Updated shape of eigen vectors: ", eigen_vectors.shape)
# After getting the eigen vectors, projecting the original data samples on a 2-D plane 

# formed by two principal eigen vectors by vector-vector multiplication

eigen_vectors.shape
standardized_data.shape
transformed_data = np.matmul(standardized_data, eigen_vectors.T)
print("Transformed data's shape: ", standardized_data.shape, " X ", eigen_vectors.T.shape, " = ", transformed_data.shape)
print(label.shape)

print(transformed_data.shape)
label = label.to_numpy().reshape(-1, 1)
labels_added = np.hstack(tup=(transformed_data, label))



df = pd.DataFrame(data=labels_added, columns=("first principal", "second principal", "digit"))

print(df.head())
import seaborn as sns

sns.FacetGrid(df, hue="digit", size=7).map(plt.scatter, "first principal", "second principal").add_legend()
from sklearn import decomposition

pca = decomposition.PCA()
pca.n_components = 2



# pca_data will contain the values of Y, the projection on principal components, hence will have transformed  data

pca_data = pca.fit_transform(sampled_data)



print("Shape of Y, transformed data: ", pca_data.shape)
# Add digit or label column

pca_data = np.hstack(tup=(pca_data, label))
df = pd.DataFrame(data=pca_data, columns=("first principal", "second principal", "digit"))
sns.FacetGrid(df, hue="digit", size=7).map(plt.scatter, "first principal", "second principal").add_legend()
pca = decomposition.PCA(n_components=784)

pca_data = pca.fit_transform(sampled_data)
percentage_var_explained = pca.explained_variance_/np.sum(pca.explained_variance_)

cum_var_explained = np.cumsum(percentage_var_explained)
plt.figure(figsize=(12, 4))

# plt.clf()

sns.set(style="whitegrid")

plt.plot(cum_var_explained, linewidth=3)

plt.axis("tight")

# plt.grid()

plt.xlabel("n_components")

plt.ylabel("cumulative explained variance")

plt.show()
from sklearn.manifold import TSNE

data1k = standardized_data[:1000, :]

label1k = label[:1000]
# The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. 

# Different values can result in significanlty different results

# n_iter is for maximumn number of iterations for the optimization

# learning rate too high and too low will n9t give proper represenation of the data

model = TSNE(n_components=2, perplexity=10, learning_rate=100, n_iter=300, verbose=2)
tsne_data = model.fit_transform(data1k)
tsne_data.shape
data = np.hstack((tsne_data, label1k))

df = pd.DataFrame(data=data, columns=("dimension 1", "dimension 2", "digit"))



sns.set(style="darkgrid")

sns.FacetGrid(df, hue="digit", size=6).map(plt.scatter, "dimension 1", "dimension 2").add_legend()
def plot_tsne_plots(data, label, perplexity, n_iter, learning_rate, data_points_to_consider, n_components):

    data1k = data[:data_points_to_consider, :]

    label1k = label[:data_points_to_consider]

    tsne_model = TSNE(n_components=n_components, n_iter=n_iter, perplexity=perplexity, learning_rate=learning_rate, verbose=2)

    tsne_data = tsne_model.fit_transform(data1k)

    

    tsne_data = np.hstack((tsne_data, label1k))

    df = pd.DataFrame(data=tsne_data, columns=("dimension 1", "dimension 2", "digit"))



    sns.set(style="darkgrid")

    sns.FacetGrid(df, hue="digit", size=6).map(plt.scatter, "dimension 1", "dimension 2").add_legend()
plot_tsne_plots(data=standardized_data, label=label, perplexity=10, n_components=2, learning_rate=100, 

                n_iter=300, data_points_to_consider=1000)
# With perplexity 50

plot_tsne_plots(data=standardized_data, label=label, perplexity=10, n_components=2, learning_rate=100, 

                n_iter=300, data_points_to_consider=1000)
# With perplexity 50

# Number of points 3000

plot_tsne_plots(data=standardized_data, label=label, perplexity=10, n_components=2, learning_rate=100, 

                n_iter=300, data_points_to_consider=3000)