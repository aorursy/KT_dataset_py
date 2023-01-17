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
dat = pd.read_csv('../input/train.csv')

lab = dat['label']

d = dat.drop('label',axis=1)

d.shape
labels = lab.head(15000)

data = d.head(15000)
from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(data)

standardized_data.shape
sample_data = standardized_data

covar_matrix = 1/15000*np.matmul(sample_data.T,sample_data)

covar_matrix.shape
from scipy.linalg import eigh



values, vectors = eigh(covar_matrix, eigvals=(782,783))

vectors.shape
import matplotlib.pyplot as plt

print(vectors.shape)

print(sample_data.T.shape)

new_coordinates = np.matmul(vectors.T, sample_data.T)

print(new_coordinates.shape)
new_coordinates = np.vstack((new_coordinates, labels)).T

print(new_coordinates)

df = pd.DataFrame(data=new_coordinates, columns=('1st principal','2nd principal','labels'))

df.head(10)
import seaborn as sn

sn.FacetGrid(df, hue="labels", size=6).map(plt.scatter, '1st principal', '2nd principal').add_legend()

plt.show()
from sklearn import decomposition

pca = decomposition.PCA()

pca.n_components = 2

pca_data = pca.fit_transform(sample_data)

pca_data.shape

pca_data = np.vstack((pca_data.T,labels)).T

pca_df = pd.DataFrame(data=pca_data,columns=('1st principal','2nd principal','labels'))

print(pca_df.shape)

pca_df.head(10)
import seaborn as sn

sn.FacetGrid(pca_df, hue="labels", size=6).map(plt.scatter, '1st principal', '2nd principal').add_legend()

plt.show()
from sklearn.manifold import TSNE

data_1000 = standardized_data[0:1000,:]

labels_1000 = labels[0:1000]
model = TSNE(n_components=2, random_state=0,perplexity=40,n_iter=5000)

tsne_data = model.fit_transform(data_1000)

print(labels.shape)

tsne_data.shape
print(labels_1000.shape)

tsne_data = np.vstack((tsne_data.T,labels_1000)).T

print(tsne_data.shape)

tsne_df = pd.DataFrame(data=tsne_data,columns=('Dim 1','Dim 2','label'))

sn.FacetGrid(tsne_df,hue='label',size=6).map(plt.scatter,'Dim 1','Dim 2').add_legend()

plt.show()