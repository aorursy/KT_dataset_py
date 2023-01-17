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

print(d0.head(5))
# Labels and Data is seggregated

l = d0['label']

d = d0.drop('label',axis = 1)

print(d.shape)

print(l.shape)
#diplay a number from the dataset

plt.figure(figsize=(7,7))

idx = 150



grid_data = d.iloc[idx].as_matrix().reshape(28,28)

plt.imshow(grid_data, interpolation = "none", cmap = "gray")



plt.show()



print(l[idx])
#Pick first 15k data-points to work on for time-efficiency.

#Exercise: Perform the same analysis on all of 42K data-point



labels = l.head(15000)

data = d.head(15000)



print("the shape of sample data = ", data.shape)
#Data-preprocessing: Standardizing the data



from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(data)

print(standardized_data.shape)
#find the co-variance matrix which is: A^T * A



sample_data = standardized_data



#matrix multiplication using numpy

covar_matrix = np.matmul(sample_data.T, sample_data)



print("The shape of variance matrix = ", covar_matrix.shape)
# finding the top two eigen-values and corresponding eigen-vectors

# projecting onto a 2-Dim space



from scipy.linalg import eigh



# the parameter 'eigvals' is defined (low value to high value)

# eigh function will return the eigen values in ascending order

# this code generate only the top 2 (782 and 783) eigenvalues.



values, vectors = eigh(covar_matrix, eigvals = (782,783))

print("Shape of eigen vectors = ",vectors.shape)



#converting the eigen vectors into (2,d) shape for the easiness of further 



vectors = vectors.T



print("Updated shape of eigen vectors =",vectors.shape)



# vectors[1] correspond to 1st principal component

# vector[2] correspond to 2nd principal component

# projecting the original data sample on the plane

# formed by two prinicpal eigen vectors by vector-vector multiplication



import matplotlib.pyplot as plt

new_coordinates = np.matmul(vectors, sample_data.T)



print("resultant new data points shape ",vectors.shape, "X", sample_data.shape)
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
#intiliazing the pca

from sklearn import decomposition

pca = decomposition.PCA()
# configruing the parameters

# the number if components = 2



pca.n_components = 2

pca_data = pca.fit_transform(sample_data)



#pca_reduced will contain the 2-d projects of the simple data

print("shape of pca_reduced.shape= ",pca_data.shape)

# attaching the label for each 2-d data point 

pca_data = np.vstack((pca_data.T, labels)).T



# creating a new data fram which help us in ploting the result data

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))

sn.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.show()
# PCA for dimensionality redcution (non-visualization)



pca.n_components = 784

pca_data = pca.fit_transform(sample_data)



# percentage variance

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

#print(percentage_var_explained)

cum_var_explained = np.cumsum(percentage_var_explained)



# Plot the PCA spectrum

plt.figure(1, figsize=(6, 4))



plt.clf()

plt.plot(cum_var_explained, linewidth=2)

plt.axis('tight')

plt.grid()

plt.xlabel('n_components')

plt.ylabel('Cumulative_explained_variance')

plt.show()





# If we take 200-dimensions, approx. 90% of variance is expalined.
from sklearn.manifold import TSNE



#picking the top 1000 point as TSNE takes a lot of time for 15k points

data_1000 = standardized_data[0:1000,:]

labels_1000 = labels[0:1000]

model = TSNE(n_components = 2, random_state = 0)

tsne_data = model.fit_transform(standardized_data)



#creating a new data frame which help us in plotting the result data

tsne_data = np.vstack((tsne_data.T, labels)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1","Dim_2","label"))



#Plotting the result of tsne

sn.FacetGrid(tsne_df, hue="label",size=6).map(plt.scatter,'Dim_1','Dim_2').add_legend()

plt.show()
model = TSNE(n_components=2, random_state=0, perplexity=50)

tsne_data = model.fit_transform(standardized_data) 



# creating a new data fram which help us in ploting the result data

tsne_data = np.vstack((tsne_data.T, labels)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))



# Ploting the result of tsne

sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.title('With perplexity = 50')

plt.show()
model = TSNE(n_components=2, random_state=0, perplexity=50,  n_iter=5000)

tsne_data = model.fit_transform(standardized_data) 



# creating a new data fram which help us in ploting the result data

tsne_data = np.vstack((tsne_data.T, labels)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))



# Ploting the result of tsne

sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.title('With perplexity = 50, n_iter=5000')

plt.show()
model = TSNE(n_components=2, random_state=0, perplexity=2)

tsne_data = model.fit_transform(standardized_data) 



# creating a new data fram which help us in ploting the result data

tsne_data = np.vstack((tsne_data.T, labels)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))



# Ploting the result of tsne

sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.title('With perplexity = 2')

plt.show()
# TSNE



from sklearn.manifold import TSNE



# Picking the top 1000 points as TSNE takes a lot of time for 15K points

data_1000 = standardized_data[0:1000,:]

labels_1000 = labels[0:1000]



model = TSNE(n_components=2, random_state=0)

# configuring the parameteres

# the number of components = 2

# default perplexity = 30

# default learning rate = 200

# default Maximum number of iterations for the optimization = 1000



tsne_data = model.fit_transform(data_1000)





# creating a new data frame which help us in ploting the result data

tsne_data = np.vstack((tsne_data.T, labels_1000)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))



# Ploting the result of tsne

sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.show()
model = TSNE(n_components=2, random_state=0, perplexity=50)

tsne_data = model.fit_transform(data_1000) 



# creating a new data fram which help us in ploting the result data

tsne_data = np.vstack((tsne_data.T, labels_1000)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))



# Ploting the result of tsne

sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.title('With perplexity = 50')

plt.show()
model = TSNE(n_components=2, random_state=0, perplexity=50,  n_iter=5000)

tsne_data = model.fit_transform(data_1000) 



# creating a new data fram which help us in ploting the result data

tsne_data = np.vstack((tsne_data.T, labels_1000)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))



# Ploting the result of tsne

sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.title('With perplexity = 50, n_iter=5000')

plt.show()
model = TSNE(n_components=2, random_state=0, perplexity=2)

tsne_data = model.fit_transform(data_1000) 



# creating a new data fram which help us in ploting the result data

tsne_data = np.vstack((tsne_data.T, labels_1000)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))



# Ploting the result of tsne

sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.title('With perplexity = 2')

plt.show()