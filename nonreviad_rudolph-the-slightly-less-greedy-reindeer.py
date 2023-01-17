import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from scipy.spatial import KDTree
df = pd.read_csv('../input/cities.csv')
data = df.as_matrix(columns=["X","Y"])
# Use KD-tree to store coordinates; makes distance queries better than O(n^2)
cities_as_kdtree = KDTree(data)
north_pole = data[0]
num_cities,_ = data.shape
# Eratosthene's sieve to determine cities' primality
primes = [True for x in range(num_cities + 1)]
primes[0] = primes[1] = False

for i in tqdm(range(2, num_cities + 1)):
    if primes[i]:
        q = i * 2
        while q <= num_cities:
            primes[q] = False
            q += i
visited = set()
curr_node = 0
total_dist = 0
path = [0]
def select_first(x):
    foo, _ = x
    return foo

for step in tqdm(range(1, num_cities + 1)):
    visited.add(curr_node)
    curr_coord = data[curr_node]
    next_node = 0
    if step < num_cities:
        # Use slightly stupid (non-optimal) strategy for looking up distances
        # Start by looking for the 2 closest neighbors...
        neighbors = 2
        if (step % 10) == 0:
            neighbors = 1000
        while True:
            distances, indices = cities_as_kdtree.query(curr_coord, k=neighbors)
            if (step % 10) == 0:
                # Make sure to add 10% to distances to non-prime cities every 10 steps
                for i in range(1, neighbors):
                    if not primes[indices[i]]:
                        distances[i] *= 1.1
                distances, indices = zip(*sorted(zip(distances,indices), key=select_first))
            # Find the closest unvisited neighbor
            for neighbor_idx in range(1, neighbors):
                neighbor = indices[neighbor_idx]
                if not (neighbor in visited):
                    next_node = neighbor
                    break
            #If all the nearest 'neighbor' neigbors have been already visited, double the number of neighbors to search. Yes, very stupid
            if next_node != 0:
                break
            neighbors *= 2
    next_coord = data[next_node]
    dist = np.linalg.norm(curr_coord - next_coord)
    if (step % 10) == 0 and not primes[next_node]:
        dist *= 1.1
    total_dist += dist
    curr_node = next_node
    path.append(curr_node)
print (total_dist)
submission = pd.DataFrame({"Path": path})
submission.to_csv("submission.csv", index=None)