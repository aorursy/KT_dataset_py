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
df = pd.read_csv(

    filepath_or_buffer='../input/iris.data.csv',

    header=None,

    sep=',')



df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

df.dropna(how="all", inplace=True) # drops the empty line at file-end



df.tail()
# split data table into data X and class labels y



X = df.ix[:,0:4].values

y = df.ix[:,4].values
from matplotlib import pyplot as plt

import numpy as np

import math



label_dict = {1: 'Iris-Setosa',

              2: 'Iris-Versicolor',

              3: 'Iris-Virgnica'}



feature_dict = {0: 'sepal length [cm]',

                1: 'sepal width [cm]',

                2: 'petal length [cm]',

                3: 'petal width [cm]'}



with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(8, 6))

    for cnt in range(4):

        plt.subplot(2, 2, cnt+1)

        for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):

            plt.hist(X[y==lab, cnt],

                     label=lab,

                     bins=10,

                     alpha=0.3,)

        plt.xlabel(feature_dict[cnt])

    plt.legend(loc='upper right', fancybox=True, fontsize=8)



    plt.tight_layout()

    plt.show()
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
import numpy as np

mean_vec = np.mean(X_std, axis=0)

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
cov_mat = np.cov(X_std.T)



eig_vals, eig_vecs = np.linalg.eig(cov_mat)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
cor_mat1 = np.corrcoef(X_std.T)



eig_vals, eig_vecs = np.linalg.eig(cor_mat1)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
cor_mat2 = np.corrcoef(X.T)



eig_vals, eig_vecs = np.linalg.eig(cor_mat2)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
u,s,v = np.linalg.svd(X_std.T)

u
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
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))



    plt.bar(range(4), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.step(range(4), cum_var_exp, where='mid',

             label='cumulative explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),

                      eig_pairs[1][1].reshape(4,1)))



print('Matrix W:\n', matrix_w)
Y = X_std.dot(matrix_w)
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),

                        ('blue', 'red', 'green')):

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

Y_sklearn = sklearn_pca.fit_transform(X_std)
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),

                        ('blue', 'red', 'green')):

        plt.scatter(Y_sklearn[y==lab, 0],

                    Y_sklearn[y==lab, 1],

                    label=lab,

                    c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='lower center')

    plt.tight_layout()

    plt.show()