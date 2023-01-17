import pandas as pd



df = pd.read_csv(

    filepath_or_buffer='../input/haberman/haberman.csv',

    header=None,

    sep=',')



df.columns=['age', 'year', 'nodes', 'status']

df.dropna(how="all", inplace=True) # drops the empty line at file-end



df.tail()
# split data table into data X and class labels y



X = df.ix[1:,0:3].values

y = df.ix[1:,3].values
from matplotlib import pyplot as plt

import numpy as np

import math



label_dict = {1: '1',

              2: '2'}



feature_dict = {0: 'age [cm]',

                1: 'year [cm]',

                2: 'nodes [cm]'}



with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(10, 8))

    for cnt in range(3):

        plt.subplot(2, 2, cnt+1)

        for lab in ('1','2'):

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



    plt.bar(range(3), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.step(range(3), cum_var_exp, where='mid',

             label='cumulative explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1),

                      eig_pairs[1][1].reshape(3,1)))



print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)



with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip(('1','2'),

                        ('black', 'green')):

        plt.scatter(Y[y==lab, 0],

                    Y[y==lab, 1],

                    label=lab,

                    c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='upper center')

    plt.tight_layout()

    plt.show()