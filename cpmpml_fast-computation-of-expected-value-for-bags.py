# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
gift_types = ['horse', 'ball', 'bike', 'train', 'coal', 'book', 'doll', 'block', 'gloves']

ngift_types = len(gift_types)

horse, ball, bike, train, coal, book, doll, block, gloves = range(ngift_types)

def gift_escore(gift, ngift, n=1000):

    # gift is the gift type

    # ngift is the number of toys in the bag

    # n is the number of sample

    if ngift == 0:

        return np.array([0.0])

    np.random.seed(2016)

    if gift == horse:

        dist = np.maximum(0, np.random.normal(5,2,(n, ngift))).sum(axis=1)

    if gift == ball:

        dist = np.maximum(0, 1 + np.random.normal(1,0.3,(n, ngift))).sum(axis=1)

    if gift == bike:

        dist = np.maximum(0, np.random.normal(20,10,(n, ngift))).sum(axis=1)

    if gift == train:

        dist = np.maximum(0, np.random.normal(10,5,(n, ngift))).sum(axis=1)

    if gift == coal:

        dist = 47 * np.random.beta(0.5,0.5,(n, ngift)).sum(axis=1)

    if gift == book:

        dist = np.random.chisquare(2,(n, ngift)).sum(axis=1)

    if gift == doll:

        dist = np.random.gamma(5,1,(n, ngift)).sum(axis=1)

    if gift == block:

        dist = np.random.triangular(5,10,20,(n, ngift)).sum(axis=1)

    if gift == gloves:

        gloves1 = 3.0 + np.random.rand(n, ngift)

        gloves2 = np.random.rand(n, ngift)

        gloves3 = np.random.rand(n, ngift)

        dist = np.where(gloves2 < 0.3, gloves1, gloves3).sum(axis=1)

    # remove bags with weight above 50

    dist = np.where(dist <= 50.0, dist, 0.0)

    return dist.mean()
epsilon = 1

max_type = np.zeros(ngift_types).astype('int')

max_value = np.zeros(ngift_types)

for gift in range(ngift_types):

    print(gift_types[gift], end=': ')

    best_value = 0.0

    for j in range(1, 100):

        value = gift_escore(gift, j)

        if value < best_value - epsilon:

            break

        best_value = value

    max_type[gift] = j

    max_value[gift] = best_value

    print(j)

    
nsample = 1000000



def weight_distributions_init(gift, ngift, n=nsample):

    #print('gift:', gift, 'ngift:', ngift)

    if ngift == 0:

        return np.array([0.0])

    np.random.seed(2016)

    if gift == horse:

        dist = np.maximum(0, np.random.normal(5,2,(n, ngift)))

    if gift == ball:

        dist = np.maximum(0, 1 + np.random.normal(1,0.3,(n, ngift)))

    if gift == bike:

        dist = np.maximum(0, np.random.normal(20,10,(n, ngift)))

    if gift == train:

        dist = np.maximum(0, np.random.normal(10,5,(n, ngift)))

    if gift == coal:

        dist = 47 * np.random.beta(0.5,0.5,(n, ngift))

    if gift == book:

        dist = np.random.chisquare(2,(n, ngift))

    if gift == doll:

        dist = np.random.gamma(5,1,(n, ngift))

    if gift == block:

        dist = np.random.triangular(5,10,20,(n, ngift))

    if gift == gloves:

        gloves1 = 3.0 + np.random.rand(n, ngift)

        gloves2 = np.random.rand(n, ngift)

        gloves3 = np.random.rand(n, ngift)

        dist = np.where(gloves2 < 0.3, gloves1, gloves3)

    for j in range(1, ngift):

        dist[:,j] += dist[:,j-1]

    return dist



all_weight_distributions = dict()

    

for gift in range(ngift_types):

    print(gift_types[gift])

    all_weight_distributions[gift] = weight_distributions_init(gift, max_type[gift])
def weight_distributions(gift, ngift):

    if ngift <= 0:

        return 0

    if ngift >= max_type[gift]:

        return 51

    return all_weight_distributions[gift][:,ngift-1]



def bagtoy_score(nballs=0, nbikes=0, nblocks=0, nbooks=0, ncoal=0, 

                           ndolls=0, ngloves=0, nhorses=0, ntrains=0):

    weights = np.zeros(nsample)

    ntypes = (nhorses, nballs, nbikes, ntrains, ncoal, nbooks, ndolls, nblocks, ngloves)

    for gift in range(ngift_types):

        weights += weight_distributions(gift, ntypes[gift])

    weights = np.where(weights <= 50.0, weights, 0.0)

    return weights.mean(), weights.std()
bagtoy_score(nballs=10)
bagtoy_score(ngloves=28)
bagtoy_score(nbikes=1, ndolls=1, ngloves=3)
bagtoy_score(nballs=6, ntrains=1, nbooks=1, ndolls=1, nblocks=1)
%timeit bagtoy_score(nballs=6, ntrains=1, nbooks=1, ndolls=1, nblocks=1)