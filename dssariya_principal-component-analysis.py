# imports



import pandas as pd

import numpy as np

%matplotlib inline

from pylab import *

from sklearn.decomposition import PCA

from itertools import cycle

# loading iris dataset



iris_dataset = pd.read_csv("../input/Iris.csv")
# shape

print(iris_dataset.shape)
# peek at the data



print(iris_dataset.head(5))
# descriptions

print(iris_dataset.describe())
# class distribution

print(iris_dataset.groupby('Species').size())
print (iris_dataset.dtypes)
# feature extraction with PCA

array = iris_dataset.values

x = array[:,1:5]

y = array[:,5]



# feature extraction



pca = PCA(n_components=2)

fit = pca.fit(x)

transform = pca.transform(x)

print(fit.components_)
print(pca.explained_variance_ratio_)

print(sum(pca.explained_variance_ratio_))