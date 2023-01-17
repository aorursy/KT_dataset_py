import math
import pandas as pd
import numpy as np
import sympy
import numba
import torch
# load
def load_cities(filename):
    cities = pd.read_csv(filename)
    city_id = cities.CityId.astype(np.int32)
    loc = np.vstack([cities.X.astype(np.float32), cities.Y.astype(np.float32)]).transpose()
    is_prime = np.array([1 if sympy.isprime(i) else 0 for i in city_id], dtype=np.int32)
    return (city_id, loc, is_prime)

def load_tour(filename):
    tour = pd.read_csv(filename)
    tour = tour.Path.values.astype(np.int32)
    return tour

@numba.jit('f4(f4[:], f4[:])', nopython=True)
def euc_2d(a, b):
    xd = a[0] - b[0]
    yd = a[1] - b[1]
    return math.sqrt(xd * xd + yd * yd)

@numba.jit('f8(i4[:], f4[:,:], i4[:])', nopython=True, parallel=True)
def cost_santa2018(tour, loc, is_prime):
    dist = 0.0
    for i in numba.prange(1, tour.shape[0]):
        a = tour[i - 1]
        b = tour[i]
        d = euc_2d(loc[a], loc[b])
        #if i % 10 == 0 and is_prime[a] == 0:
        #    d *= 1.1
        dist += d
    return dist
city_id, loc, is_prime = load_cities("../input/cities.csv")
# create a random tour
tour = np.array(city_id + [0], dtype=np.int32)
cost_santa2018(tour, loc, is_prime)
%timeit cost_santa2018(tour, loc, is_prime)
tloc = torch.tensor(loc, dtype=torch.float64).cuda()
tprime = torch.zeros(tour.shape[0]-1, dtype=torch.float64).cuda()

def cost_santa2018_torch(tour, loc, is_prime):
    ttour = torch.tensor(tour, dtype=torch.long).cuda()
    tprime[9::10] = torch.tensor((1 - is_prime[tour[9:-1:10]])*.1)
    dist = (tloc[ttour][1:] - tloc[ttour][:-1]).pow(2).sum(1).sqrt_()
    return dist.sum().data.tolist()
    return (dist.sum() + (dist * tprime).sum()).data.tolist()
cost_santa2018_torch(tour, loc, is_prime)
%timeit cost_santa2018_torch(tour, loc, is_prime)
