import pandas as pd



#Handle the numerical arrays

import numpy as np 



#Plotting the data

import seaborn as sns

import matplotlib.pyplot as plt



#Download MNIST dataset

from sklearn import datasets

from sklearn import manifold



%matplotlib inline
#Feth data from SKlearn

data = datasets.fetch_openml(

    'mnist_784',

    version=1,

    return_X_y=True

)

features, target = data 
features
# size of each image is 28X28

features.shape
target
target = target.astype(int)

target
target.shape
features[1,:].shape
# Selecting first row and rehape into 28X28

img = features[0,:].reshape(28,28)
plt.imshow(img, cmap = 'gray')
# tsne used for dimensionality reduction and used the technique of similarity score

tsne = manifold.TSNE(

    n_components=2,

    random_state=42

)
tsne_data = tsne.fit_transform(features[:2000,])
tsne_data.shape
df = pd.DataFrame(np.column_stack((tsne_data, target[:2000])), columns = ['X','Y','output'])
df.head()
df.shape
df.dtypes
grid = sns.FacetGrid(df, hue = 'output', size= 8)

grid.map(plt.scatter, 'X', 'Y').add_legend()