from concorde.tsp import TSPSolver
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import time
cities = pd.read_csv('../input/cities.csv')
cities_scaled = cities.copy()
cities_scaled.iloc[:, 1:] *= 1000
solver = TSPSolver.from_data(
    cities_scaled.X,
    cities_scaled.Y,
    norm="EUC_2D"
)

t = time.time()
tour_data = solver.solve(time_bound = 60.0, verbose = True, random_seed = 42) # solve() doesn't seem to respect time_bound for certain values?
print(time.time() - t)
print(tour_data.found_tour)

# function to get simple numbers
def primes(n):
    a = [0] * n 
    for i in range(n):
        a[i] = i 
        
    a[1] = 0
     
    m = 2 
    while m < n: 
        if a[m] != 0: 
            j = m * 2
            while j < n:
                a[j] = 0
                j = j + m
        m += 1
    
    b = []
    for i in a:
        if a[i] != 0:
            b.append(a[i])
    
    del a
    return b
# route length calculation
def route_len(tour_data, cities):
    route = tour_data.tour
    prime_nums = primes(route.shape[0])
    coords = cities.iloc[route, 1:].values
    coord_diff = coords[1:] - coords[:-1]
    coeffs = np.ones((coord_diff.shape[0], ))
    for idx in tqdm_notebook(range(9, coord_diff.shape[0], 10)):
        if route[idx] not in prime_nums:
            coeffs[idx] += 0.1
    length = np.sum(np.sqrt(np.power(coord_diff[:, 0], 2) + np.power(coord_diff[:, 1], 2)) *  coeffs.T)
    return length
length = route_len(tour_data, cities)
length_scaled = route_len(tour_data, cities_scaled)
print('Length: {}, Scaled length: {}'.format(length, length_scaled))
pd.DataFrame({'Path': np.append(tour_data.tour,[0])}).to_csv('submission_scaled.csv', index=False)