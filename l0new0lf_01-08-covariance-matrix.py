import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# datset

from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)



np.random.seed(123)
# Boston Dataset

m, n = X.shape

print(f"num of samples: {m}\nnum of feats: {n}")
# return covariance_matrix

def get_covar_matrix(dataset_X, is_col_standardized=False):

    

    if type(dataset_X) != np.ndarray: raise Exception("X must be a numpy array")

    

    if is_col_standardized is False:

        # column standardize `dataset_X` (w/ matrix operations)

        n_means   = np.mean(X, axis=0)

        n_stds    = np.std(X, axis=0)

        dataset_X = (X-n_means) / (n_stds) # automatic broadcasting

    

    # use col-standardized formula

    num_samples, _ = dataset_X.shape

    covar_mat = (1/num_samples) * (dataset_X.T.dot(dataset_X))

    

    return covar_mat
covar_mat = get_covar_matrix(X, is_col_standardized=False)

covar_mat.shape
# dont use matshow

plt.imshow(covar_mat)



plt.xlabel("feature indies")

plt.ylabel("feature indies")

plt.title("Covariance matrix")

plt.colorbar()

plt.show()
# 1. STANDARDIZE

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()



X_std = standardizer.fit_transform(X)

cov_mat = np.cov(X_std.T)
plt.figure(figsize=(10,10))

sns.set(font_scale=1)



labels = [f"feat{i}" for i in range(0, 13)]



hm = sns.heatmap(

        cov_mat,

        cbar=True,

        annot=True,

        square=True,

        fmt='.2f',

        annot_kws={'size': 7},

        yticklabels= labels,

        xticklabels= labels,

        cmap='viridis'

    )





plt.title('Covariance matrix showing correlation coefficients\n')

plt.show()