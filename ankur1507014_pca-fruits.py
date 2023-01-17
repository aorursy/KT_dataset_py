import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



fruits = pd.read_table('../input/IPythonData_04042018.txt')

X_fruits = fruits[['mass','width','height', 'color_score']]

y_fruits = fruits['fruit_name']
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit(X_fruits).transform(X_fruits)  
mean_vec = np.mean(X_std, axis=0)

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)

print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
eig_vals, eig_vecs = np.linalg.eig(cov_mat)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
cor_mat1 = np.corrcoef(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
cor_mat2 = np.corrcoef(X_fruits.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
u,s,v = np.linalg.svd(X_std.T)

u

for ev in eig_vecs.T:

    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

print('Everything ok!')
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(key=lambda x: x[0], reverse=True)

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
Y.shape


with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip(('apple', 'mandarin', 'orange', 'lemon'),

                        ('blue', 'red', 'green', 'pink')):

        plt.scatter(Y[y_fruits==lab, 0],

                    Y[y_fruits==lab, 1],

                    label=lab,

                    c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='lower right')

    plt.tight_layout()

    plt.show()
from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)

Y_sklearn = sklearn_pca.fit_transform(X_std)



with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip(('apple', 'mandarin', 'orange', 'lemon'),

                        ('blue', 'red', 'green', 'pink')):

        plt.scatter(Y[y_fruits==lab, 0],

                    Y[y_fruits==lab, 1],

                    label=lab,

                    c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='lower right')

    plt.tight_layout()

    plt.show()