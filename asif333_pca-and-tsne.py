import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

 

d0 = pd.read_csv("../input/Iris.csv")

    



print(d0.head(5)) 



# save the labels into a

labels = d0['Species']



# Drop the label feature and store the pixel data in d.

data = d0.drop("Species",axis=1)

data=data.drop("Id",axis=1)

print(data.head())
from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(data)

print(standardized_data.shape)
sample_data = standardized_data



# matrix multiplication using numpy

covar_matrix = np.matmul(sample_data.T , sample_data)



print ( "The shape of variance matrix = ", covar_matrix.shape)

from scipy.linalg import eigh 



# the parameter 'eigvals' is defined (low value to heigh value) 

# eigh function will return the eigen values in asending order

# this code generates only the top 2 (782 and 783) eigenvalues.

values, vectors = eigh(covar_matrix, eigvals=(2,3))



print("Shape of eigen vectors = ",vectors.shape)

# converting the eigen vectors into (2,d) shape for easyness of further computations

vectors = vectors.T



print("Updated shape of eigen vectors = ",vectors.shape)

# here the vectors[1] represent the eigen vector corresponding 1st principal eigen vector

# here the vectors[0] represent the eigen vector corresponding 2nd principal eigen vector
import matplotlib.pyplot as plt

new_coordinates = np.matmul(vectors, sample_data.T)



print (" resultanat new data points' shape ", vectors.shape, "X", sample_data.T.shape," = ", new_coordinates.shape)
import pandas as pd

import numpy as np



# appending label to the 2d projected data

new_coordinates = np.vstack((new_coordinates, labels)).T



# creating a new data frame for ploting the labeled points.

dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))

print(dataframe.head())
import seaborn as sn

sn.FacetGrid(dataframe, hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.show()
from sklearn import decomposition

pca = decomposition.PCA()

pca.n_components = 2

pca_data = pca.fit_transform(sample_data)



# pca_reduced will contain the 2-d projects of simple data

print("shape of pca_reduced.shape = ", pca_data.shape)



pca_data = np.vstack((pca_data.T, labels)).T



# creating a new data fram which help us in ploting the result data

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))

sn.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.show()
pca.n_components = 4

pca_data = pca.fit_transform(sample_data)



percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);



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

from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)

# configuring the parameteres

# the number of components = 2

# default perplexity = 30

# default learning rate = 200

# default Maximum number of iterations for the optimization = 1000



tsne_data = model.fit_transform(data)





# creating a new data frame which help us in ploting the result data

tsne_data = np.vstack((tsne_data.T, labels)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))



# Ploting the result of tsne

sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.show()
model = TSNE(n_components=2, random_state=0, perplexity=30)

tsne_data = model.fit_transform(data) 



# creating a new data fram which help us in ploting the result data

tsne_data = np.vstack((tsne_data.T, labels)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))



# Ploting the result of tsne

sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.title('With perplexity = 50')

plt.show()
model = TSNE(n_components=2, random_state=0, perplexity=50,  n_iter=5000)

tsne_data = model.fit_transform(data) 



# creating a new data fram which help us in ploting the result data

tsne_data = np.vstack((tsne_data.T, labels)).T

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))



# Ploting the result of tsne

sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.title('With perplexity = 50, n_iter=5000')

plt.show()