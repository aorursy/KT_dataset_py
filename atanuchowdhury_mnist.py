# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d0=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
print(d0.head())  # print first five rows of d0.



# save the labels into a variable l.
l=d0['label']

# Drop the label feature and store the pixel data in d.
d=d0.drop('label',axis=1)
print(d.shape)
print(l.shape)
# display or plot a number.
plt.figure(figsize=(7,7))
idx=500

grid_data=d.iloc[idx].to_numpy().reshape(28,28) # reshape from 1d to 2d pixel array
plt.imshow(grid_data,interpolation='none',cmap='gray')
plt.show()

print(l[idx])
labels=l.copy()
data=d.copy()

print("The shape of the data",data.shape)
# Data-preprocessing: Standardizing the data

from sklearn.preprocessing import StandardScaler
standardized_data=StandardScaler().fit_transform(data)
print(standardized_data.shape)
#find the co-variance matrix which is : A^T * A
sample_data=standardized_data

# Matrix multiplication using numpy
covar_matrix=np.matmul(sample_data.T,sample_data)

print("The shape of the variance matrix = ",covar_matrix.shape)
# finding the top two eigen-values and corresponding eigen-vectors 
# for projecting onto a 2-Dim space.

from scipy.linalg import eigh

# the parameter 'eigvals' is defined (low value to high value) 
# eigh function will return the eigen values in asending order
# this code generates only the top 2 (782 and 783) eigenvalues.
values,vectors=eigh(covar_matrix,eigvals=(782,783))

print("Shape of eigen vectors = ",vectors.shape)

# converting the eigen vectors into (2,d) shape for easyness of further computations
vectors=vectors.T
print("updated shape of eigen vectors =",vectors.shape)

# here the vectors[1] represent the eigen vector corresponding 1st principal eigen vector
# here the vectors[0] represent the eigen vector corresponding 2nd principal eigen vector

# projecting the original data sample on the plane 
#formed by two principal eigen vectors by vector-vector multiplication.
import matplotlib.pyplot as plt
new_coordinates=np.matmul(vectors,sample_data.T)

print (" resultant new data points' shape ", vectors.shape, "X", sample_data.T.shape," = ", new_coordinates.shape)
import pandas as pd

# appending label to the 2d projected data
new_coordinates=np.vstack((new_coordinates,labels)).T


# creating a new data frame for ploting the labeled points.
dataframe=pd.DataFrame(new_coordinates,columns=("1st_principle","2nd_principle","label"))
print(dataframe.head())
# ploting the 2d data points with seaborn
import seaborn as sns
sns.FacetGrid(dataframe,hue="label",size=8).map(plt.scatter,'1st_principle','2nd_principle').add_legend()
plt.show()
# Initializing the pca
from sklearn import decomposition
pca=decomposition.PCA()
# configuring the parameteres
# the number of components = 2
pca.n_components=2
pca_data=pca.fit_transform(sample_data)

# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced data= ",pca_data.shape)
# attaching the label for each 2-d data point 
pca_data = np.vstack((pca_data.T, labels)).T

# creating a new data fram which help us in ploting the result data
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))
sns.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()
