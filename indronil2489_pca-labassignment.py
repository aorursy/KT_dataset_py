# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Any results you write to the current directory are saved as output.
df = pd.read_csv(

    filepath_or_buffer='../input/wine.csv',header=None,sep=',')



df.columns=['Wine', 'Alcohol', 'Malic.acid', 'Ash', 'Acl', 'Mg', 'Phenols','Flav', 'Nf.phenols', 'Pronath','Color.int', 'Hue', 'OD', 'Proline']

df.dropna(how="all", inplace=True) # drops the empty line at file-end



df.tail()
X = df.ix[1:,1:14].values

print(X)

y = df.ix[1:,0].values

print(y)
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
import numpy as np

mean_vec = np.mean(X_std, axis=0)

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)
cov_mat = np.cov(X_std.T)



eig_vals, eig_vecs = np.linalg.eig(cov_mat)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
u,s,v = np.linalg.svd(X_std.T)

u
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
from matplotlib import pyplot as plt

import numpy as np

import math



with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(10, 6))



    plt.bar(range(13), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.step(range(13), cum_var_exp, where='mid',

             label='cumulative explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
matrix_w = np.hstack((eig_pairs[0][1].reshape(13,1),

                      eig_pairs[1][1].reshape(13,1)))



print('Matrix W:\n', matrix_w)
Y = X_std.dot(matrix_w)
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 5))

    for lab, col in zip(('1', '2', '3'),

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

    plt.figure(figsize=(6, 5))

    for lab, col in zip(('1', '2', '3'),

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