import numpy as np

import numpy.linalg

import matplotlib.pyplot as plt

%matplotlib inline
sample_order_2 = [np.random.randn(2,2) for _ in range(1000)]

ranks_order_2 = [np.linalg.matrix_rank(M) for M in sample_order_2]

plt.xticks(range(4))

plt.hist(ranks_order_2, range(4))
np.linalg.matrix_rank(np.random.randn(2,2,2))
import numpy as np

import numpy.linalg

from functools import reduce

import string

import pylab as plt
## Adapted from https://stackoverflow.com/a/13772838/160466

def parafac(factors):

    ndims = len(factors)

    request=''

    for temp_dim in range(ndims):

        request+=string.ascii_lowercase[temp_dim]+'z,'

    request=request[:-1]+'->'+string.ascii_lowercase[:ndims]

    return np.einsum(request,*factors)
## Alternating Least Squares algorithm to compute a CP decomposition.

## Taken from Kolda & Bader 2009, Figure 8.

## Coded by nic 20171210. All credit is theirs, mistakes are mine!

def cp_als(X, R, maxiter=1000, test_period=1, tol=1e-10):

    A = [np.random.randn(In, R) for In in X.shape]

    N = len(X.shape)

    for iteration in range(maxiter):

        for n in range(N):

            non_n = [m for m in range(N) if m != n]

            non_An = [A[m] for m in non_n]

            V = reduce(np.multiply, map((lambda a: np.dot(a.T, a)), non_An))

            W = reduce((lambda b, c: np.einsum('ij, kj -> ikj', b, c).reshape(-1, b.shape[1])), non_An)

            Xn = np.rollaxis(X, n).reshape(X.shape[n], -1)

            newAn = np.dot(Xn, np.dot(W, np.linalg.pinv(V)))

            lam = np.linalg.norm(A[n], axis=0)

            A[n] = newAn / lam



        if iteration % test_period==0:

            lA = [np.dot(A[0], np.diag(lam))] + A[1:]

            Xhat = parafac(lA)

            if np.linalg.norm(X - Xhat) < tol:

                break

    return lam, A, iteration
X = np.random.randn(2,2,2)

lam, A, iteration = cp_als(X, 3)

print("Ended in {} iterations.".format(iteration))



print(lam)

for An in A:

    print(An)



A[0] = np.dot(A[0], np.diag(lam))



Xhat = parafac(A)



print("INPUT")

print(X)

    

print("PARAFAC")

print(Xhat)



print("Error: {}".format(np.linalg.norm(X-Xhat)))
def find_error(X, R):

    lam, A, its = cp_als(X, R)

    A[0] = np.dot(A[0], np.diag(lam))

    error = np.linalg.norm(X - parafac(A))

    return error



def find_rank(X):

    error = [find_error(X, R) for R in range(1, 4)]

    rank = sum(np.array(error)>1e-10)+1

    return rank
sample_order_3 = [np.random.randn(2,2,2) for _ in range(100)]

ranks_order_3 = [find_rank(M) for M in sample_order_3]
ranks_order_3[:20]
plt.xticks(range(4))

plt.hist(ranks_order_3, range(5))