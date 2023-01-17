# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/digit-recognizer/train.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df.head(10)
# save the labels into l
l = df['label']

# store the pixel into d, except l
d = df.drop("label", axis=1)
# d
# check shape
print(l.shape)
print(d.shape)
import matplotlib.pyplot as plt

# display number
plt.figure(figsize=(7,7))
idx=1

# reshape pixel from 1d to 2d pixel array
grid = d.iloc[idx].to_numpy().reshape(28, 28)
plt.imshow(grid, interpolation="none", cmap="gray")
plt.show()

print(l[idx])
labels = l.head(42000)
data = d.head(42000)

print("shape of sample data =" , data.shape)
# Standardizing the data
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)
print(standardized_data.shape)
# Find Co-varience matrix A^T * A

sample_data = standardized_data

covar_mat = np.matmul(sample_data.T, sample_data)
print('shape of varience matrix =', covar_mat.shape)
# finding the top two eigen-values and corresponding eigen-vectors 
# for projecting onto a 2-Dim space.
from scipy.linalg import eigh

values, vectors = eigh(covar_mat, eigvals=(782, 783))
vectors = vectors.T

print('shape of eigen vectors = ', vectors.shape)

new_cor = np.matmul(vectors, sample_data.T)
print('resultant new data points shape', vectors.shape, "X", 
                             sample_data.T.shape, " = ", new_cor.shape)
# appending label to 2D projected data
new_cor = np.vstack((new_cor, labels)).T

# create new DF for plotting labelled points
dataF = pd.DataFrame(data=new_cor, columns=("1st_principal", "2nd_principal", "label"))
dataF
#print(dataF.head())
import seaborn as sn
sn.FacetGrid(dataF, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()
from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca_data = pca.fit_transform(sample_data)

pca_data.shape
# attach label for each 2-d dataPoints
pca_data = np.vstack((pca_data.T, labels)).T

pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))
sn.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()
pca.n_components=784
pca_data = pca.fit_transform(sample_data)

per_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)

cum_var_explained = np.cumsum(per_var_explained)

# plot PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_varience')
plt.show()
