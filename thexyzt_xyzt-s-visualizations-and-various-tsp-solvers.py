%matplotlib inline
import numpy as np
import pandas as pd
from sympy import isprime, primerange
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from concorde.tsp import TSPSolver
import time
cities = pd.read_csv('../input/cities.csv')
cities['isPrime'] = cities.CityId.apply(isprime)
prime_cities = cities.loc[cities.isPrime]
plt.figure(figsize=(16,10))
plt.subplot(111, adjustable='box', aspect=1.0)
plt.plot(cities.X, cities.Y, 'k,', alpha=0.3)
plt.plot(cities.X[0], cities.Y[0], 'bx')
plt.xlim(0, 5100)
plt.ylim(0, 3400)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('All cities (North Pole = Blue X)', fontsize=18)
plt.show()
plt.figure(figsize=(16,10))
plt.subplot(111, adjustable='box', aspect=1.0)
plt.plot(cities.X, cities.Y, 'k,', alpha=0.3)
plt.plot(prime_cities.X, prime_cities.Y, 'r.', markersize=4, alpha=0.3)
plt.plot(cities.X[0], cities.Y[0], 'bx')
plt.xlim(0, 5100)
plt.ylim(0, 3400)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('All cities (Primes = Red Dots, North Pole = Blue X)', fontsize=18)
plt.show()
# This function will submit a path to name.csv (with some validity tests)
def make_submission(name, path):
    assert path[0] == path[-1] == 0
    assert len(set(path)) == len(path) - 1 == 197769
    pd.DataFrame({'Path': path}).to_csv(f'{name}.csv', index=False)

# Fast score calculator given a path
def score_path(path):
    cities = pd.read_csv('../input/cities.csv', index_col=['CityId'])
    pnums = [i for i in primerange(0, 197770)]
    path_df = cities.reindex(path).reset_index()
    
    path_df['step'] = np.sqrt((path_df.X - path_df.X.shift())**2 + 
                              (path_df.Y - path_df.Y.shift())**2)
    path_df['step_adj'] = np.where((path_df.index) % 10 != 0,
                                   path_df.step,
                                   path_df.step + 
                                   path_df.step*0.1*(~path_df.CityId.shift().isin(pnums)))
    return path_df.step_adj.sum()
def nearest_neighbour():
    cities = pd.read_csv("../input/cities.csv")
    ids = cities.CityId.values[1:]
    xy = np.array([cities.X.values, cities.Y.values]).T[1:]
    path = [0,]
    while len(ids) > 0:
        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]
        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
    path.append(0)
    make_submission('nearest_neighbour', path)
    return path

#path_nn = nearest_neighbour()
def concorde_tsp(seed=42):
    cities = pd.read_csv('../input/cities.csv')
    solver = TSPSolver.from_data(cities.X, cities.Y, norm="EUC_2D")
    tour_data = solver.solve(time_bound=60.0, verbose=True, random_seed=seed)
    if tour_data.found_tour:
        path = np.append(tour_data.tour,[0])
        make_submission('concorde', path)
        return path
    else:
        return None

path_cc = concorde_tsp()
cities = pd.read_csv('../input/cities.csv')
cities['isPrime'] = cities.CityId.apply(isprime)
prime_cities = cities.loc[(cities.CityId == 0) | (cities.isPrime)]
solver = TSPSolver.from_data(prime_cities.X, prime_cities.Y, norm="EUC_2D")
tour_data = solver.solve(time_bound=5.0, verbose=True, random_seed=42)
prime_path = np.append(tour_data.tour,[0])
plt.figure(figsize=(16,10))
ax = plt.subplot(111, adjustable='box', aspect=1.0)
ax.plot(cities.X, cities.Y, 'k,', alpha=0.3)

lines = [[(prime_cities.X.values[prime_path[i]],
           prime_cities.Y.values[prime_path[i]]),
          (prime_cities.X.values[prime_path[i+1]],
           prime_cities.Y.values[prime_path[i+1]])]
         for i in range(0, len(prime_cities))]
lc = mc.LineCollection(lines, linewidths=1, colors='r')
ax.add_collection(lc)

ax.plot(cities.X[0], cities.Y[0], 'bx')
plt.xlim(0, 5100)
plt.ylim(0, 3400)

plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('All cities (Prime Path = Red, North Pole = Blue X)', fontsize=18)
plt.show()