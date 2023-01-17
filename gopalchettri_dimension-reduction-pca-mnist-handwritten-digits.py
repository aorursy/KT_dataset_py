

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

dataset =pd.read_csv('../input/mnist_train.csv')

print(dataset.head(5))
# save the labels into a variable named l

l = dataset['label']

# drop the label feature and store the pixel data in d

d = dataset.drop('label', axis=1)
print(d.shape)

print(l.shape)
# display or plot a number

plt.figure(figsize=(7,7))

idx=50



# reshape from 1d to 2d pixel array

grid_data = d.iloc[idx].values.reshape(28,28)

plt.imshow(grid_data, interpolation='none', cmap='gray')

plt.show()



# print index value of the label

print(l[idx])
# 2D visualization using PCA

labels = l

data = d

print("The shape of sample data = ", data.shape)
# Standardizing the data

# standardizing means calculating the Z score

from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(data)

print(standardized_data.shape)
# find the covariance matrix which is : A^T * A

sample_data = standardized_data



#matrix multiplication using numpy

covar_matrix = np.matmul(sample_data.T, sample_data)

print("The shape of variance matrix = ", covar_matrix.shape)
# finding the top two eigen-values and corresponding eigen-vectors

# for projecting onto a 2-Dimensional Space

from scipy.linalg import eigh



# the parameter 'eigvals' is defined (low value to high value)

# eigh function will return the eigen values in ascending order

# this code generates only the top 2 (783 and 782) eigenvalues.

values, vectors = eigh(covar_matrix, eigvals=(782, 783))



print("Shape of eigen vectors = ", vectors.shape)



# converting the eigen vectors into 2d shape for easyness of further computation

vectors = vectors.T



print("Updated shape of eigen vectors = ", vectors.shape)

# here the vectors[1] represent the eigen vector corresponding 1st principal eigen vector

# here the vectors[0] represent the eigen vector corresponding 2nd principal eigen vector
# projecting the original data sample on the plane

# formed by two principal eigen vectors by vector-vector mulitplication

new_coordinates = np.matmul(vectors, sample_data.T)

print ("Resultanat new data points' shape ", vectors.shape, "X", sample_data.T.shape," = ", new_coordinates.shape)
# appending label to the 2nd projected data

new_coordinates = np.vstack((new_coordinates, labels)).T



# creating a new data frame for plotting the labeled points

dataframe = pd.DataFrame(data=new_coordinates, columns=('1st_principal', '2nd_principal', 'label'))

print(dataframe.head())
# ploting the 2d data points with seaborn

import seaborn as sn

sn.FacetGrid(dataframe, hue="label", height=8).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.show()
from sklearn.decomposition import PCA

# initializing the pca

pca = PCA(n_components=2) # configuring the number of components

pca_data = pca.fit_transform(sample_data)
# pca_data will contain the 2d projects of simple data

print("Shape of pca_reduced.shape = ", pca_data.shape)
# attaching the label for each 2-d data point

pca_data = np.vstack((pca_data.T, labels)).T



# creating a new data frame which help us in ploting the result data

pca_df = pd.DataFrame(data=pca_data, columns=('1st_principal', '2nd_principal', 'label'))

sn.FacetGrid(pca_df, hue='label', height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.show()