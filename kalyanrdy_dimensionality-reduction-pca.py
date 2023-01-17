# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
d0 = pd.read_csv("../input/train.csv")
print(d0.head(5)) # print first five rows of d0.

# save the labels into a variable l.
l = d0['label']

# Drop the label feature and store the pixel data in d.
d = d0.drop("label",axis=1)
plt.figure(figsize=(7,7))
idx = 1

grid_data = d.iloc[idx].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array
plt.imshow(grid_data, interpolation = "none", cmap = "gray")
plt.show()

print(l[idx])
labels = l 
data = d 

from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)

sample_data = standardized_data
covar_matrix = np.matmul(sample_data.T, sample_data)

from scipy.linalg import eigh

values, vectors = eigh(covar_matrix, eigvals=(782,783))
vectors = vectors.T

new_coordinates = np.matmul(vectors, sample_data.T)
new_coordinates = np.vstack((new_coordinates,labels)).T
dataframe = pd.DataFrame(data=new_coordinates, columns=("1st","2nd","label"))
print(dataframe.head())
import seaborn as sns

sns.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st', '2nd').add_legend()
plt.show()
from sklearn import decomposition
pca = decomposition.PCA()
pca.n_components = 2
pca_data = pca.fit_transform(sample_data)

