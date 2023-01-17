import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import time

from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import cdist, pdist
cities = pd.read_csv("../input/cities.csv")
available_nodes = set(cities.CityId.values)
positions = np.array([cities.X.values, cities.Y.values]).T
hull = ConvexHull(positions)

# Finding the two most distant positions on the hull
mat = distance_matrix(positions[hull.vertices], positions[hull.vertices])
i, j = np.unravel_index(mat.argmax(), mat.shape)

# Initializing our tour, and removing those cities from the set of available nodes
tour =  np.array([hull.vertices[i], hull.vertices[j]])
available_nodes.remove(hull.vertices[i])
available_nodes.remove(hull.vertices[j])
nodes_arr = np.ma.masked_array([i for i in available_nodes])
best_distances = np.ma.masked_array(cdist(positions[nodes_arr], positions[tour], 'euclidean').min(axis=1))

# We want the most distant node, so we get the max
index_to_remove = best_distances.argmax()
next_id = nodes_arr[index_to_remove]

# Add the most distant point, as well as the first point to close the tour, we'll be inserting from here
tour = np.append(tour, [next_id, tour[0]])

available_nodes.remove(next_id)
nodes_arr[index_to_remove] = np.ma.masked
best_distances[index_to_remove] = np.ma.masked
# Takes two arrays of points and returns the array of distances
def dist_arr(x1, x2):
    return np.sqrt(((x1 - x2)**2).sum(axis=1))

# This is our selection method we will be using, it will give us the index in the masked array of the selected node,
# the city id of the selected node, and the updated distance array.
def get_next_insertion_node(nodes, positions, prev_id, best_distances):
    best_distances = np.minimum(cdist(positions[nodes], positions[prev_id].reshape(-1, 2), 'euclidean').min(axis=1), best_distances)
    max_index = best_distances.argmax()
    return max_index, nodes[max_index], best_distances
start_time = time.time()
progress = 3
while len(available_nodes) > 0:
    index_to_remove, next_id, best_distances = get_next_insertion_node(nodes_arr, positions, next_id, best_distances)
    progress += 1
    
    # Finding the insertion point
    c_ik = cdist(positions[tour[:-1]], positions[next_id].reshape(-1, 2))
    c_jk = cdist(positions[tour[1:]], positions[next_id].reshape(-1, 2))
    c_ij = dist_arr(positions[tour[:-1]],positions[tour[1:]]).reshape(-1, 1)
    i = (c_ik + c_jk - c_ij).argmin()
    
    tour = np.insert(tour, i+1, next_id)

    available_nodes.remove(next_id)
    nodes_arr[index_to_remove] = np.ma.masked
    best_distances[index_to_remove] = np.ma.masked
    
    if progress % 1000 == 0:
        print(f'Progress: {progress}, Remaining: {len(available_nodes)}')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(tour)
tour = np.delete(tour, -1)
tour = np.roll(tour, -tour.argmin())
tour = np.append(tour, 0)
print(tour)
plt.figure(figsize=(20,20))
plt.title("Farthest Insertion Method")
plt.plot(*zip(*positions[tour]), '-r')
plt.scatter(*zip(*positions), c="b", s=10, marker="s")
plt.show()
def dist_1d(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def score_tour(tour, positions):
    primes = set(sym.sieve.primerange(1, tour.max()+1))
    score = 0
    for i, (j, k) in enumerate(zip(tour[:-1], tour[1:])):
        score += dist_1d(positions[j], positions[k]) * (1.1 if (i+1) % 10 == 0 and j not in primes else 1)
    return score

score_tour(tour, positions)
pd.DataFrame(data={'Path':tour}).to_csv('submission.csv', index=False)