import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv("../input/train.csv")

data.head(5)
a = data["label"].unique()

a.sort()

print("There are {} unique label values present in the given dataset, which are {}".format(a.shape[0], a))
labels = data["label"]
data_final = data.drop("label",axis = 1)
print("The shape of the input data is {}, which means that there are {} number of entries ".format(data_final.shape, data_final.shape[0]))

print ("The number of the train labels are {}".format(labels.shape[0]))
## Lets plot the image to check the input data (SANITY CHECK)

plt.figure(figsize =(7,7))

idx = 121



grid_data = data_final.iloc[idx].as_matrix().reshape(28,28)

plt.imshow(grid_data, interpolation = "none", cmap = "gray")

plt.show()



print(labels[idx])
from sklearn.preprocessing import StandardScaler



standardised_data = StandardScaler().fit_transform(data_final)

print(standardised_data.shape)
covariance_matrix = (1/42000)*(np.matmul(standardised_data.T, standardised_data))

print("The shape of our Covariance Matrix is {}".format(covariance_matrix.shape))
from scipy.linalg import eigh



values,vectors = eigh(covariance_matrix, eigvals=(782,783))

print(vectors.shape)



vectors = vectors.T
print(vectors.shape)
new_coordinates = np.matmul(vectors, standardised_data.T)

print( "new data points are", vectors.shape, "X" ,standardised_data.T.shape, "is", new_coordinates.shape)
new_coordinates = np.vstack((new_coordinates, labels)).T

dataframe = pd.DataFrame(data = new_coordinates, columns = ("1st_Principal", "2nd_Principal", "label"))

print(dataframe.head(5))
import seaborn as sns



sns.FacetGrid(dataframe, hue = "label", size = 10).map(plt.scatter, "1st_Principal", "2nd_Principal").add_legend()

plt.show()
from sklearn import decomposition

pca = decomposition.PCA()

pca.n_components = 2 

pca_data = pca.fit_transform(standardised_data)

print(pca_data.shape)
pca_data = np.vstack((pca_data.T, labels)).T

pca_data.shape
pca_data_final = pd.DataFrame(data = pca_data, columns = ("1st_Principal", "2nd_Principal", "labels"))

sns.FacetGrid(pca_data_final, hue = "labels", size = 10).map(plt.scatter, "1st_Principal", "2nd_Principal").add_legend()

plt.show()
from sklearn import decomposition 

pca = decomposition.PCA()

pca.ncomponents = 784

pca_data_2 = pca.fit_transform(standardised_data)

percentage_var_explained = pca.explained_variance_/np.sum(pca.explained_variance_)

cum_var_explained = np.cumsum(percentage_var_explained)

plt.figure(1, figsize = (10,6))

plt.clf()

plt.plot(cum_var_explained, linewidth = 2)

plt.axis("tight")

plt.grid()

plt.xlabel("n_components")

plt.ylabel("Cumulative Explained Variance")

plt.show()
from sklearn.manifold import TSNE



data_5000 = standardised_data[0:5000,:]

labels_5000 = labels[0:5000]
model = TSNE(n_components = 2, perplexity = 50.0, random_state = 0)

tsne_data = model.fit_transform(data_5000)
tsne_data = np.vstack((tsne_data.T,labels_5000)).T

tsne_df = pd.DataFrame(tsne_data, columns = ("1st_Dim","2nd_Dim","Labels"))
sns.FacetGrid(tsne_df, hue = "Labels", size = 10).map(plt.scatter, "1st_Dim", "2nd_Dim").add_legend()

plt.show()