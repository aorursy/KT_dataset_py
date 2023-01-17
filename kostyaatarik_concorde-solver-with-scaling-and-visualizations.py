import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concorde.tsp import TSPSolver
from sympy import isprime, primerange
primes = {i for i in primerange(0, 197770)}
cities = pd.read_csv('../input/cities.csv', index_col=['CityId'])

XY = np.stack((cities.X.values, cities.Y.values), axis=1)
add = np.array([0 if i in primes else 0.1 for i in cities.index])

def score_path(path):
    xy = XY[path, :]
    steps = np.sqrt(
        np.sum(
            np.square(xy - np.roll(xy, -1, axis=0)),
            axis=1))
    return steps.sum() + (steps[9::10] * add[path[9::10]]).sum()
# Instantiate solver
solver = TSPSolver.from_data(
    cities.X * 10000,
    cities.Y * 10000,
    norm="CEIL_2D"  # more exact in couple with scaling
)
import random

best_score, best_path = 2 * 10**6, None
for i in range(5):
    tour_data = solver.solve(time_bound = 60.0, verbose = True, random_seed=random.randint(0, 2019))
    path = np.append(tour_data.tour,[0])
    if score_path(path[::-1]) < score_path(path):
        path = path[::-1]
    score = score_path(path)
    if score < best_score:
        best_score, best_path = score, path
path = best_path
pd.DataFrame({'Path': path}).to_csv('submission.csv', index=False)
%matplotlib inline
plt.figure(figsize=(32,18))
plt.plot(cities.X[path], cities.Y[path])
plt.show()
prime_path, composite_path = [0], []
for i in path:
    if i in primes:
        prime_path.append(i)
    else:
        composite_path.append(i)
prime_path.append(0)

plt.figure(figsize=(32,18))
plt.plot(cities.X[composite_path], cities.Y[composite_path])
plt.plot(cities.X[prime_path], cities.Y[prime_path])
plt.show()

plt.figure(figsize=(32,18))
plt.plot(cities.X[composite_path], cities.Y[composite_path])
plt.show()

plt.figure(figsize=(32,18))
plt.plot(cities.X[prime_path], cities.Y[prime_path], color='red')
plt.show()
n_good, n_bad = 0, 0
good_edges, bad_edges = [], []
for i, city_id in enumerate(path[:-1], 1):
    if i % 10 == 0:
        if city_id in primes:
            n_good += 1
            good_edges.append([path[i-1], path[i]])
        else:
            n_bad += 1
            bad_edges.append([path[i-1], path[i]])
print(f'Number of "good" tenth moves: {n_good}, number of "bad" tenth moves: {n_bad}.')
# # please make it faster with LineCollection
# plt.figure(figsize=(32,18))
# for edge in good_edges:
#     plt.plot(cities.X[edge], cities.Y[edge], color='green')
# for edge in bad_edges:
#     plt.plot(cities.X[edge], cities.Y[edge], color='red')
# plt.show()
def pure_score(path):
    xy = XY[path, :]
    steps = np.sqrt(
        np.sum(
            np.square(xy - np.roll(xy, -1, axis=0)),
            axis=1))
    return steps.sum()

print("Pure path's length is shorter by {:.2f}.".format(score_path(path) - pure_score(path)))