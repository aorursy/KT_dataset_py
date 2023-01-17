# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv('../input/heart-disease-uci/heart.csv', header=None)

df.head()
x = df.ix[1:,0:12].values

y = df.ix[1:,13].values

x
from sklearn.preprocessing import StandardScaler

x_std = StandardScaler().fit_transform(x)
x_std
x_std.shape[0]
import numpy as np

mean_vec = np.mean(x_std, axis=0)

cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec)) / (x_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)
cov_mat = np.cov(x_std.T)



eig_vals, eig_vecs = np.linalg.eig(cov_mat)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
for ev in eig_vecs.T:

    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

print('Everything ok!')
# Make a list of (eigenvalue, eigenvector) tuples

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

eig_pairs.sort(key=lambda x: x[0], reverse=True)



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:')

for i in eig_pairs:

    print(i[0])
tot = sum(eig_vals)

var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)
cum_var_exp
eig_pairs
matrix_w = np.hstack((eig_pairs[0][1].reshape(13,1),

                      eig_pairs[1][1].reshape(13,1)))



print('Matrix W:\n', matrix_w)
Y = x_std.dot(matrix_w)

Y
y
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(12, 6))

    for lab, col in zip(('1','0'),

                        ('red', 'green')):

        plt.scatter(Y[y==lab, 0],

                    Y[y==lab, 1],

                    label=lab,

                    c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='lower center')

    plt.tight_layout()

    plt.show()
from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)

Y_sklearn = sklearn_pca.fit_transform(x_std)
Y_sklearn
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip(('1', '0'),

                        ('red', 'green')):

        plt.scatter(Y_sklearn[y==lab,0],

                    Y_sklearn[y==lab,1],

                    label=lab,

                    c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='lower center')

    plt.tight_layout()

    plt.show()