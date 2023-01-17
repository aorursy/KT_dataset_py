%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np
yeast = "http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"

col_names = ["sequence name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "target"]



yeast_df = pd.read_csv(yeast, delim_whitespace=True, lineterminator='\n', names=col_names)

yeast_df.head()
def center_matrix(data):

    ctr_matrix = data.copy(deep=True)

    

    for column in ctr_matrix:

        mean = np.mean(data[column])

        

        # Center the Matrix

        for i in range(data.shape[0]):

            ctr_matrix.loc[i, column] -= mean

            

    return ctr_matrix



def fraction_of_variance (eigen_vals, alpha):

    total_variance = sum(eigen_vals)

    for i in range(len(eigen_vals)):

        fr = sum(eigen_vals[:i+1]) / total_variance

        

        if fr >= alpha:   # Choose Dimensionality

            return i

            

def pca_numpy(data, alpha):

    n,m = data.shape

     

    Z = center_matrix(data)     # Center the Matrix

    C = np.dot(Z.T, Z) / (n)    # Compute Covariance Matrix

    

    # Eigen decomposition

    eigen_vals, eigen_vecs = np.linalg.eig(C)

    

    # Fraction of Total Variance

    r = fraction_of_variance (eigen_vals, alpha)

    

    # Reduced Basis

    eigen_vecs = eigen_vecs[:r+1]

    print("Eigen Value =", eigen_vals)

    

    # Reduced Dimensionality Data

    numpy_pca = np.dot(data, eigen_vecs.T)

    

    return numpy_pca, eigen_vals
features = ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"]



pca_manual, eig_vals_numpy = pca_numpy(yeast_df[features], 0.92)

r_numpy = pca_manual.shape[1]





pca_manual = pd.DataFrame(data = pca_manual)

pca_manual
from sklearn.decomposition import PCA



data = yeast_df.loc[:, features]



sklearn_pca = PCA()

sklearn_pca = sklearn_pca.fit(data)



eig_vals_sklearn = sklearn_pca.explained_variance_

print("Eigen Value =", eig_vals_sklearn)

r_sklearn = fraction_of_variance (eig_vals_sklearn, 0.92) + 1

print(r_sklearn)



sklearn_pca = sklearn_pca.transform(data)

sklearn_pca = pd.DataFrame(data=sklearn_pca)

sklearn_pca
def MSE (eig_vals, r):

    return np.sum(eig_vals) - np.sum(eig_vals[:r])



mse_pca_numpy = MSE(eig_vals_numpy, r_numpy)

print("MSE soal nomor satu:", mse_pca_numpy)



mse_pca_sklearn = MSE(eig_vals_sklearn, r_sklearn)

print("MSE soal nomor dua: ", mse_pca_sklearn)
