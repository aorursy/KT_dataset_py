import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def innerproduct(X,Z=None):

    # function innerproduct(X,Z)

    #

    # Computes the inner-product matrix.

    # Syntax:

    # D=innerproduct(X,Z)

    # Input:

    # X: nxd data matrix with n vectors (rows) of dimensionality d

    # Z: mxd data matrix with m vectors (rows) of dimensionality d

    #

    # Output:

    # Matrix G of size nxm

    # G[i,j] is the inner-product between vectors X[i,:] and Z[j,:]

    #

    # call with only one input:

    # innerproduct(X)=innerproduct(X,X)

    #

    if Z is None: # case when there is only one input (X)

        Z=X;

    G = np.dot(X,Z.T)

    return(G)
def l2distance(X,Z=None):

    # function D=l2distance(X,Z)

    #

    # Computes the Euclidean distance matrix.

    # Syntax:

    # D=l2distance(X,Z)

    # Input:

    # X: nxd data matrix with n vectors (rows) of dimensionality d

    # Z: mxd data matrix with m vectors (rows) of dimensionality d

    #

    # Output:

    # Matrix D of size nxm

    # D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)

    #

    # call with only one input:

    # l2distance(X)=l2distance(X,X)

    #

    if Z is None:

        Z=X;



    n,d1=X.shape

    m,d2=Z.shape

    assert (d1==d2), "Dimensions of input vectors must match!"

    

    if n!=m :

        dists = -2 * innerproduct(X,Z)

        dists += np.sum(X**2, axis=1)[:, np.newaxis]

        dists += np.sum(Z**2, axis=1)

    else:

        dists = - 2 * innerproduct(X,Z)

        dists += np.sum(X**2,axis=1)[:, np.newaxis]    

        dists += np.sum(Z**2,axis=1) ;

    

    D = np.sqrt(np.abs(np.nan_to_num(dists))) ####Sometimes very small numvers can be non-negative due to numerical precision, you can just set them to 0.

                               

   

   # D = np.na_to_num(D)



    return(D)
import time

current_time = lambda: int(round(time.time() * 1000))



X=np.random.rand(700,100)

Z=np.random.rand(300,100)



print("Running the vectorized version...")

before = current_time()

Dfast=l2distance(X)

after = current_time()

t_fast = after - before

print("{:2.0f} ms".format(t_fast))
