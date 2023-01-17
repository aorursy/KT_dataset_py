import matplotlib.pyplot as plt 

import numpy as np 

import pandas as pd 

import seaborn as sns 

from sklearn import datasets # this library is used to read the dataset provided by sklearn for practice purpose

from sklearn import manifold # this sklearn library is used for t-sne

%matplotlib inline
data = datasets.fetch_openml( 'mnist_784',#this data i m iporting from sklearn

                             version=1,

                             return_X_y=True ) # here i m reading both predictor and target columns 
pixel_values, targets = data # here i m retrieving both columns and target , where pixel_values are predictor columns 

targets = targets.astype(int) #converting target value to Integer type
print(targets.shape)

print(pixel_values.shape)
#here, i am ploting the 3rd pic and i am reshaping the 3rd row of data set which is 1*784 array we are reshaping 28*28 

single_image1 = pixel_values[3, :].reshape(28, 28) 

plt.imshow(single_image1, cmap='gray')
# here,i am creating instance of 2 components of t-SNE only from 784 columns  of data 

tsne = manifold.TSNE(n_components=2, random_state=42)

transformed_data = tsne.fit_transform(pixel_values[:1000, :]) #here, for simlicity reading only 1000 rows of data and converting into 
data_frame_tsne = pd.DataFrame( np.column_stack((transformed_data, targets[:1000])), columns=["x", "y", "targets"] )

data_frame_tsne.loc[:, "targets"] = data_frame_tsne.targets.astype(int)
data_frame_tsne
grid = sns.FacetGrid(data_frame_tsne, hue="targets", size=8) 

grid.map(plt.scatter, "x", "y").add_legend()
from sklearn.decomposition import PCA
pca = PCA(n_components=2,random_state=42)

pca_transformed_data = pca.fit_transform(pixel_values[:1000, :])
data_frame_pca = pd.DataFrame( np.column_stack((pca_transformed_data, targets[:1000])), columns=["x", "y", "targets"] )

data_frame_pca.loc[:, "targets"] = data_frame_pca.targets.astype(int)
data_frame_pca
grid = sns.FacetGrid(data_frame_pca, hue="targets", size=8) 

grid.map(plt.scatter, "x", "y").add_legend()