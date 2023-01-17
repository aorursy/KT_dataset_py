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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
d0 = pd.read_csv('../input/train.csv')



print(d0.head(5)) # print first five rows of d0.



# save the labels into a variable l.

l = d0['label']



# Drop the label feature and store the pixel data in d.

d = d0.drop("label",axis=1)
# display or plot a number.

plt.figure(figsize=(7,7))

idx = 1



grid_data = d.iloc[idx].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array

plt.imshow(grid_data, interpolation = "none", cmap = "gray")

plt.show()



print(l[idx])
# Pick first 15K data-points to work on for time-effeciency.





labels = l.head(15000)

data = d.head(15000)



print("the shape of sample data = ", data.shape)
# Data-preprocessing: Standardizing the data

#mean =0 and STD=1

from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(data)

print(standardized_data.shape)
#find the co-variance matrix which is : A^T * A

sample_data = standardized_data



# matrix multiplication using numpy

covar_matrix = np.matmul(sample_data.T , sample_data)



print ( "The shape of variance matrix = ", covar_matrix.shape)
# finding the top two eigen-values and corresponding eigen-vectors 

# for projecting onto a 2-Dim space.



from scipy.linalg import eigh 



# the parameter 'eigvals' is defined (low value to heigh value) 

# eigh function will return the eigen values in asending order

# this code generates only the top 2 (782 and 783) eigenvalues.

values, vectors = eigh(covar_matrix, eigvals=(782,783))



print("Shape of eigen vectors = ",vectors.shape)

# converting the eigen vectors into (2,d) shape for easyness of further computations

vectors = vectors.T



print("Updated shape of eigen vectors = ",vectors.shape)

# here the vectors[1] represent the eigen vector corresponding 1st principal eigen vector

# here the vectors[0] represent the eigen vector corresponding 2nd principal eigen vector
# projecting the original data sample on the plane 

#formed by two principal eigen vectors by vector-vector multiplication.



import matplotlib.pyplot as plt

new_coordinates = np.matmul(vectors, sample_data.T)



print (" resultanat new data points' shape ", vectors.shape, "*", sample_data.T.shape," = ", new_coordinates.shape)
import pandas as pd



# appending label to the 2d projected data

new_coordinates = np.vstack((new_coordinates, labels)).T



# creating a new data frame for ploting the labeled points.

dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))

print(dataframe.head())
# ploting the 2d data points with seaborn

import seaborn as sn

sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.show()
# initializing the pca

from sklearn import decomposition

pca = decomposition.PCA()
# configuring the parameteres

# the number of components = 2

pca.n_components = 2

pca_data = pca.fit_transform(sample_data)



# pca_reduced will contain the 2-d projects of simple data

print("shape of pca_reduced.shape = ", pca_data.shape)
# attaching the label for each 2-d data point 

pca_data = np.vstack((pca_data.T, labels)).T



# creating a new data fram which help us in ploting the result data

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))

sn.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.show()