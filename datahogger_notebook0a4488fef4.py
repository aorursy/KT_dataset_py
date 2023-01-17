# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print("Are there NaNs in the dataset?: ",train.isnull().values.any())
binary = []

cate = []

for c in train:

    if c == 'ID' or c == 'y':

        pass

    else:

        if train[[c]].isin([0,1]).all().values :

            binary.append(c)

        else:

            cate.append(c)
print("Number of Binary features      ",len(binary))

print("Number of Categorical features ",len(cate))
qq = [ np.unique(train[[x]].values) for x in cate]

xq = [i for xqq in qq for i in xqq] # flattening

w = np.unique(xq,return_counts=True)

if np.all(w[1] == 1):

    print("All of the entries in categorical features are unique.")

else:

    print("There is a repetition of entries across the categorical features.")
sr = []

for x in binary:

    try :

        sr.append(spearmanr(train[[x]].values,train[['y']])[0])

    except:

        sr.append(0)
plt.figure(figsize=(40,20))

plt.plot(sr,'-r',lw=1)

plt.grid(True)

plt.xticks(np.arange(len(sr)),binary,rotation=90)

plt.show()
from sklearn.neighbors import NearestNeighbors

from scipy.special import gamma,psi

from scipy import ndimage

from scipy.linalg import det

from numpy import pi
def MI_DC(x, y, k):

    """

    Calculates the mututal information between a continuous vector x and a

    disrete class vector y.

    This implementation can calculate the MI between the joint distribution of

    one or more continuous variables (X[:, 1:3]) with a discrete variable (y).

    Thanks to Adam Pocock, the author of the FEAST package for the idea.

    Brian C. Ross, 2014, PLOS ONE

    Mutual Information between Discrete and Continuous Data Sets

    """

    y = y.flatten()

    n = x.shape[0]

    classes = np.unique(y)

    knn = NearestNeighbors(n_neighbors=k)

    # distance to kth in-class neighbour

    d2k = np.empty(n)

    # number of points within each point's class

    Nx = []

    for yi in y:

        Nx.append(np.sum(y == yi))



    # find the distance of the kth in-class point

    for c in classes:

        mask = np.where(y == c)[0]

        knn.fit(x[mask, :])

        d2k[mask] = knn.kneighbors()[0][:, -1]



    # find the number of points within the distance of the kth in-class point

    knn.fit(x)

    m = knn.radius_neighbors(radius=d2k, return_distance=False)

    m = [i.shape[0] for i in m]



    # calculate MI based on Equation 2 in Ross 2014

    MI = psi(n) - np.mean(psi(Nx)) + psi(k) - np.mean(psi(m))

    return MI





def MI_CC(variables, k=1):

    """

    Returns the mutual information between any number of variables.

    Here it is used to estimate MI between continuous X(s) and y.

    Written by Gael Varoquaux:

    https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429

    """



    all_vars = np.hstack(variables)

    return (sum([Entropy(X, k=k) for X in variables]) -

            Entropy(all_vars, k=k))





def Nearest_Distance(X, k=1):

    '''

    X = array(N,M)

    N = number of points

    M = number of dimensions

    returns the distance to the kth nearest neighbor for every point in X

    '''

    knn = NearestNeighbors(n_neighbors=k)

    knn.fit(X)

    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself

    return d[:, -1] # returns the distance to the kth nearest neighbor



def Entropy(X, k=1):

    ''' Returns the entropy of the X.

    Parameters

    ===========

    X : array-like, shape (n_samples, n_features)

        The data the entropy of which is computed

    k : int, optional

        number of nearest neighbors for density estimation

    Notes

    ======

    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy

    of a random vector. Probl. Inf. Transm. 23, 95-101.

    See also: Evans, D. 2008 A computationally efficient estimator for

    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.

    and:

    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual

    information. Phys Rev E 69(6 Pt 2):066138.

    '''



    # Distance to kth nearest neighbor

    r = Nearest_Distance(X, k) # squared distances

    n, d = X.shape

#     volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)

    ge = .5*d + 1

    lv = 0.5*d*np.log(pi) - .5*np.log(2*pi*ge) - ge*np.log(ge) + ge - d*np.log(2)

    '''

    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures

    for Continuous Random Variables. Advances in Neural Information

    Processing Systems 21 (NIPS). Vancouver (Canada), December.

    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)

    '''

    return (d*np.mean(np.log(r + np.finfo(np.float).eps))

            + lv + psi(n) - psi(k))





def MI(variables, k=1):

    '''

    Returns the mutual information between any number of variables.

    Each variable is a matrix X = array(n_samples, n_features)

    where

      n = number of samples

      dx,dy = number of dimensions

    Optionally, the following keyword argument can be specified:

      k = number of nearest neighbors for density estimation

    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)

    '''

    if len(variables) < 2:

        raise AttributeError(

                "Mutual information must involve at least 2 variables")

    all_vars = np.hstack(variables)

    return (sum([Entropy(X, k=k) for X in variables])

            - Entropy(all_vars, k=k))
midc = []

for x in binary:

    try :

        z = MI_CC([train[['y']].values,train[[x]].values],3)

        midc.append(z)

    except:

        print("x",x)

        midc.append(0)
plt.figure(figsize=(40,20))

plt.plot(midc,'-r',lw=1)

plt.grid(True)

plt.xticks(np.arange(len(midc)),binary,rotation=90)

plt.show()