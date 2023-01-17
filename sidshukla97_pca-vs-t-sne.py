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
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
train.shape
test.head()
test.shape
# save the labels into a variable
label = train['label']

# Drop the label feature and store the pixel data
data = train.drop("label", axis=1)
print(data.shape)
print(label.shape)
# display or plot a number
import matplotlib.pyplot as plt
%matplotlib inline


plt.figure(figsize=(7,7))
idx = 1

grid_data = data.iloc[idx].as_matrix().reshape(28, 28)
plt.imshow(grid_data, interpolation="none", cmap="gray")
plt.show()

print(label[idx])
#before doing dimensionality reduction we must perform data-preprocessing for better understandable format
#We can utilise Eigenvalues and Eigenvectors to reduce the dimension space
# Data-preprocessing: Standardizing the data

from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)
print(standardized_data.shape)
#find the co-variance matrix which is : A^T * A
sample_data = standardized_data

# matrix multiplication using numpy
covar_matrix = np.matmul(sample_data.T , sample_data)


print("the shape of co-variance matrix= ", covar_matrix.shape)

#eigenvector does not change direction in a transformation
# finding the top two eigen-values and corresponding eigen-vectors 
# for projecting onto a 2-Dim space.

from scipy.linalg import eigh


# the parameter 'eigvals' is defined (low value to heigh value) 
# eigh function will return the eigen values in asending order
#top eigen values 

values, vectors = eigh(covar_matrix, eigvals=(782, 783))


print("Shape of eigen vectors = ",vectors.shape)
# converting the eigen vectors into (2,d) shape for easyness of further computations
vectors = vectors.T

print("Updated shape of eigen vectors = ",vectors.shape)
# here the vectors[1] represent the eigen vector corresponding 1st principal eigen vector
# here the vectors[0] represent the eigen vector corresponding 2nd principal eigen vector
# projecting the original data  on the plane 
#formed by two principal eigen vectors by vector-vector multiplication.

import matplotlib.pyplot as plt
new_coordinates = np.matmul(vectors, sample_data.T)

print (" resultanat new data points' shape ", vectors.shape, "X", sample_data.T.shape," = ", new_coordinates.shape)
import pandas as pd

#appending label to the 2nd projected data
new_coordinates = np.vstack((new_coordinates, label)).T

#creating a new data frame for ploting the labeled points
dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))
print(dataframe.head())
#ploting the 2d data points with seaborn
import seaborn as sns
sns.FacetGrid(dataframe, hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()
#overlapping is lot and pca does not do good job
#tsne does a lot better 
#initializing the pca

from sklearn import decomposition
pca = decomposition.PCA()
#configuring the parameters
#the number of components = 2

pca.n_components = 2
pca_data=  pca.fit_transform(sample_data)

#pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape", pca_data.shape)
import warnings
warnings.filterwarnings("ignore")
# attaching the label for each 2-d data point 
pca_data = np.vstack((pca_data.T, label)).T

# creating a new data fram which help us in ploting the result data
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))
sns.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()
#just as same its kinda slighlty rotated 90 degrees
#pca for dimensionality reduction (not for visulaization)

pca.n_components = 784
pca_data = pca.fit_transform(sample_data)

#nothing but eigen values
percentage_var_explained = pca.explained_variance_/ np.sum(pca.explained_variance_)
cum_var_explained = np.cumsum(percentage_var_explained)

#plot the pca spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('cumulative_explained_variance')
plt.show()

#if we take 200-dimesions , approx . 90% of variance is explained
from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)
# configuring the parameteres
# the number of components = 2

tsne_data=  model.fit_transform(sample_data)

tsne_data = np.vstack((tsne_data.T, label)).T
tsne_df = pd.DataFrame(tsne_data, columns = ("Dim_1", "Dim_2", "label"))

#plotting the result of tsne
sns.FacetGrid(tsne_df, hue='label', size=6).map(plt.scatter, "Dim_1", 'Dim_2').add_legend()
plt.show()
