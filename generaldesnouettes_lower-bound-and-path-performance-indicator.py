# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sympy import isprime, primerange
from math import sqrt
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from matplotlib.path import Path
# For busy visualizations
plt.rcParams['agg.path.chunksize'] = 10000

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Loading cities and defining primes
cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv', index_col=['CityId'])
XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)
is_prime = np.array([1 if isprime(i) else 0 for i in cities.index], dtype=np.int32)
# Using a KD Tree to identify nearest neighbours:
kdt = KDTree(XY)
# Find 3 nearest neighbors (including city itself)
dists, neibs = kdt.query(XY, 3)
# List of neighbours
neibs
# List of distances
dists
## Lower bound per city
arr_LB = 0.5 * (dists[:, 1] + dists[:, 2])
## Lower path distance
path_LB_score = np.sum(arr_LB)
print('Theoretical Lower Bound path would score {}.'.format(path_LB_score))
# Loading a path from public kernels as an example
path = np.array(pd.read_csv('../input/dp-shuffle-by-blacksix/DP_Shuffle.csv').Path)
# Because I am not very efficient and piecing up parts of different kernels together I will reload primes
# Load the prime numbers we need in a set with the Sieve of Eratosthenes
def eratosthenes(n):
    P = [True for i in range(n+1)]
    P[0], P[1] = False, False
    p = 2
    l = np.sqrt(n)
    while p < l:
        if P[p]:
            for i in range(2*p, n+1, p):
                P[i] = False
        p += 1
    return P

def load_primes(n):
    return set(np.argwhere(eratosthenes(n)).flatten())

PRIMES = load_primes(cities.shape[0])
# Running the list of distances in & out for each city in the path (as well as overall score to double check)
coord = cities[['X', 'Y']].values
score = 0
arr_perfo = np.copy(arr_LB)
for i in range(1, len(path)):
    begin = path[i-1]
    end = path[i]
    distance = np.linalg.norm(coord[end] - coord[begin])
    if i%10 == 0:
        if begin not in PRIMES:
            distance *= 1.1
    score += distance
    arr_perfo[begin] -= distance/2
    arr_perfo[end] -= distance/2
print('Path score: {}.'.format(score))
# This gives a list of "inefficiencies" per city
arr_perfo
# Difference between "Lower Bound" path and current path
np.sum(arr_perfo)
sq_perfo = arr_perfo * arr_perfo
## Scatter Plot
cities.plot.scatter(x='X', y='Y', s=sq_perfo , figsize=(15, 10), c=sq_perfo, cmap='Reds' )
north_pole = cities.iloc[0]
plt.scatter(north_pole.X, north_pole.Y, c='red', s=15)
plt.axis('off')
plt.show()
# Loading a path from public kernels as an example
path = np.array(pd.read_csv('../input/lkh-solver-by-aguiar/LKH_Solver.csv').Path)
score = 0
arr_perfo = np.copy(arr_LB)

for i in range(1, len(path)):
    begin = path[i-1]
    end = path[i]
    distance = np.linalg.norm(coord[end] - coord[begin])
    if i%10 == 0:
        if begin not in PRIMES:
            distance *= 1.1
    score += distance
    arr_perfo[begin] -= distance/2
    arr_perfo[end] -= distance/2
print('Path score: {}.'.format(score))

sq_perfo = arr_perfo * arr_perfo

## Scatter Plot
cities.plot.scatter(x='X', y='Y', s=sq_perfo , figsize=(15, 10), c=sq_perfo, cmap='Reds' )
north_pole = cities.iloc[0]
plt.scatter(north_pole.X, north_pole.Y, c='red', s=15)
plt.axis('off')
plt.show()


