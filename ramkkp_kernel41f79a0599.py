# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#d0 = pd.read_csv('./mnist_train.csv')

train = pd.read_csv("../input/mnist-digit-recognizer/train.csv")

print(train.head(5)) # print first five rows of d0.

train.shape

# save the labels into a variable l.

l = train['label']

print(l.shape)

# Drop the label feature and store the pixel data in d.

d = train.drop("label",axis=1)

print(d.shape)
# display or plot a number.

plt.figure(figsize=(7,7))

idx = 1



grid_data = d.iloc[idx].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array

plt.imshow(grid_data, interpolation = "none", cmap = "gray")

plt.show()



print(l[idx])
# Pick first 15K data-points to work on for time-effeciency.

#Excercise: Perform the same analysis on all of 42K data-points.



label = l.head(15000)

data = d.head(15000)



print("the shape of sample data = ", data.shape)
#Sample for Standardscaler example:

    

from sklearn.preprocessing import StandardScaler

data_tes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])



scaler = StandardScaler().fit_transform(data_tes)

print(data_tes)

print(scaler)

#scaler.mean(axis=0)

scaler.std(axis = 0)
# Data-preprocessing: Standardizing the data



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

print (" resultanat new data points' shape ", vectors.shape, "X", sample_data.T.shape," = ", new_coordinates.shape)





import pandas as pd



# appending label to the 2d projected data

new_coordinates = np.vstack((new_coordinates, label)).T



# creating a new data frame for ploting the labeled points.

dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))

print(dataframe.head())
# ploting the 2d data points with seaborn

import seaborn as sns



sns.FacetGrid(dataframe,hue = 'label', size = 6).map(plt.scatter,"1st_principal", "2nd_principal").add_legend()
from sklearn import decomposition

pca = decomposition.PCA()

pca



pca.n_components=2

pca_data = pca.fit_transform(sample_data)

print("shape of PCA_reduced.shape=",pca_data.shape)
pca_data1 = np.vstack((pca_data.T,label.T)).T

pca_data1

pca_data2 = pd.DataFrame(pca_data1,columns=("1st_principal", "2nd_principal","label"))

pca_data2



sns.FacetGrid(pca_data2,hue='label',size=6).map(plt.scatter,"1st_principal", "2nd_principal").add_legend()

#importing the Dataset

train_test = pd.read_csv("../input/mnist-digit-recognizer/train.csv")

train_test.shape

#extracting the Data and label part of the dataset(before sampling)

l_tes = train_test['label']

d_tes = train_test.drop('label',axis=1)

l_tes

d_tes
#Sampling the data(30k)

l_samp =  l_tes.head(30000)

d_samp = d_tes.head(30000)

d_samp.shape
#Normalization of the sample data

from sklearn.preprocessing import StandardScaler



d_stnd = StandardScaler().fit_transform(d_samp)



#finding the covanrance using matrix multiplication

mat_dst = np.matmul(d_stnd,d_stnd.T)

mat_dst.shape
#finding the hiehest eigen value and eighen vectors

from scipy.linalg import eigh



values, vectors = eigh(mat_dst,eigvals = (782,783))

vectors.shape()

vectors = vectors.T
new_co = np.matmul(vectors,d_samp.T)

d_samp.shape()



#appending a label to the 2d projected data

new_col = pd.vstack((new_co,l_samp)).T



#Creating a new dataframe out of array new_col

new_codf = pd.DataFrame(new_col,columns=("1st_principal", "2nd_principal","label"))



sns.FacetGrid(new_codf,hue='label',size =6).map(plt.scatter,"1st_principal", "2nd_principal").add_legend()
