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
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.DESCR)
X = iris.data

y = iris.target
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)

sns.distplot(X[y==0, 0], label="Setosa")

sns.distplot(X[y==1, 0], label="Versicolor")

sns.distplot(X[y==2, 0], label="Virginica")

plt.xlabel("Sepal Length[cm]", fontsize=15)

plt.subplot(2, 2, 2)

sns.distplot(X[y==0, 1], label="Setosa")

sns.distplot(X[y==1, 1], label="Versicolor")

sns.distplot(X[y==2, 1], label="Virginica")

plt.xlabel("Sepal Width[cm]", fontsize=15)

plt.subplot(2, 2, 3)

sns.distplot(X[y==0, 2], label="Setosa")

sns.distplot(X[y==1, 2], label="Versicolor")

sns.distplot(X[y==2, 2], label="Virginica")

plt.xlabel("Petal Length[cm]", fontsize=15)

plt.subplot(2, 2, 4)

sns.distplot(X[y==0, 3], label="Setosa")

sns.distplot(X[y==1, 3], label="Versicolor")

sns.distplot(X[y==2, 3], label="Virginica")

plt.xlabel("Petal Width[cm]", fontsize=15)

plt.legend()
# Above is the distribution of all4 features accroding to class labels.

# Which features have the high variance or explain the data most, let's find out.
# We need to standardize the data 

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
# Mean of each feature is very very close to zero

np.mean(X_std, axis=0)
# Calculate the covariance matrix 

mean_vec = np.mean(X_std, axis=0)

cov_mat = ((X_std - mean_vec).T.dot(X_std - mean_vec))/(X_std.shape[0]-1)
cov_mat
# Now let's find out the eigen vectors and eigen values of our covariance matrix

eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
# Eigenvalues

eigen_values
# Eigenvectors

# Each column in below matrix is a eigen vector

eigen_vectors
# Let's make a pair value of eigen vectors and eigen values

eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]

eigen_pairs
# Let's sort these pairs based on eigen values

eigen_pairs.sort(key=lambda x: x[0], reverse=True)
for pair in eigen_pairs:

    print(pair[0])
# Explained variance 

explained_var = (eigen_values/eigen_values.sum())*100

explained_var
# So we take these two principal components only 

W = np.concatenate([eigen_pairs[0][1].reshape(4, 1), eigen_pairs[1][1].reshape(4, 1)], axis=1)
print("Matrix W:\n", W)
Y = X_std.dot(W)
Y.shape
sns.set(style="darkgrid")

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)

sns.scatterplot(Y[y==0, 0], Y[y==0, 1], label="Setosa", s=60)

sns.scatterplot(Y[y==1, 0], Y[y==1, 1], label="Versicolor", s=60)

sns.scatterplot(Y[y==2, 0], Y[y==2, 1], label="Virginica", s=60)

plt.xlabel("Component 1")

plt.ylabel("Component 2")

plt.subplot(1, 2, 2)

sns.scatterplot(X[y==0, 2], X[y==0, 3], label="Setosa", s=60)

sns.scatterplot(X[y==1, 2], X[y==1, 3], label="Versicolor", s=60)

sns.scatterplot(X[y==2, 2], X[y==2, 3], label="Virginica", s=60)

plt.xlabel("Petal Length")

plt.ylabel("Petal Width")

plt.legend()