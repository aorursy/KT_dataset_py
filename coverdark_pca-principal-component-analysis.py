import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
def pca_custom(X, k):

    n, p = np.shape(X)

    # make Z with mean zero

    Z = X - np.mean(X, axis=0)

    # calculate covariance maxtrix

    covZ = np.cov(Z.T)

    # calculate eigenvalues and eigenvectors

    eigValue, eigVector = np.linalg.eig(covZ)

    # sort eigenvalues in descending order

    index = np.argsort(-eigValue)

    

    # select k principle components

    if k > p:

        print ("k must lower than input data dimension！")

        return 1

    else:

        # select k eigenvectors with largest eigenvalues

        selectVector = eigVector[:, index[:k]]

        T = np.matmul(Z, selectVector)

    return T, selectVector
use_sklearn = True



# generate data

# x = np.random.randn(100, 20)



y = np.random.randn(20, 1)

x = np.matmul(y, [[1.3, -0.5]])



# set k

k = 2

# PCA

pcaX, selectVector = pca_custom(x, k)



print('PCA transformation matrix: ')

print(selectVector)



if use_sklearn:

    z = x - np.mean(x, axis=0)

    pcaResults = PCA(n_components=k).fit(z)

    print ('PCA transformation matrix from sklearn:')

    print(pcaResults.components_.T)

    newX = np.matmul(z, pcaResults.components_.T)



print('original data')

plt.plot(x[:, 0], x[:, 1], 'ok')

plt.show()



print('After PCA')

plt.plot(pcaX[:, 0], pcaX[:, 1], 'or')

plt.show()



if use_sklearn:

    print('After PCA from sklearn')

    plt.plot(newX[:, 0], newX[:, 1], 'ob')

    plt.show()