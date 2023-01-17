import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

df = pd.read_csv('../input/cities.csv')
city_coordinates = df[['X','Y']].values
num_cities,_ = city_coordinates.shape
# Eratosthene's sieve to determine cities' primality
primes = [True for x in range(num_cities + 1)]
primes[0] = primes[1] = False
for i in tqdm_notebook(range(2, num_cities + 1)):
    if primes[i]:
        q = i * 2
        while q <= num_cities:
            primes[q] = False
            q += i
primes = np.asarray(primes)
not_primes = np.asarray([not x for x in primes])

mask_tens = [.1 if (x % 10) == 9 else 0 for x in range(num_cities)]
def create_best_first():
    north_pole = np.array(city_coordinates[0])

    current_city_coords = np.array(city_coordinates[0])
    num_cities = city_coordinates.shape[0]

    # keep the list of unvisited cities
    unvisited_cities = [x for x in range(1, num_cities)]
    unvisited_coordinates = np.array(city_coordinates[1:]) # actually remove the startup city from the coordinates 2D array

    unvisited = num_cities - 1

    path = []
    total_distance = 0

    #trace the route backwards (by step number)
    for step in tqdm_notebook(range(num_cities, 1, -1)):
        # unvisited_coordinates[:unvisited] contains for each row the coordinates of an unvisited city 
        # (unvisited_coordinates[X] has the coordinates of unvisited_cities[X])
        distances = np.linalg.norm(unvisited_coordinates[:unvisited] - current_city_coords, axis=1)

        if step % 10 == 0:
            distances += np.multiply(distances, not_primes[unvisited_cities[:unvisited]]) * 0.1

        closest_city_index = np.argmin(distances)
        closest_city = unvisited_cities[closest_city_index]
        current_city_coords = np.array(unvisited_coordinates[closest_city_index]) 

        total_distance += distances[closest_city_index]
        path.append(closest_city)

        # "Remove" closest_city from the two lists
        unvisited_coordinates[closest_city_index] = unvisited_coordinates[unvisited - 1]
        unvisited_cities[closest_city_index] = unvisited_cities[unvisited - 1]
        
        unvisited -= 1
    
    path = [0] + path[::-1] + [0] # reverse the path and add the North Pole at both ends
    last_dist = np.linalg.norm(north_pole - current_city_coords)
    total_distance += last_dist
    return (path, total_distance)
    
best_path, score = create_best_first()
print ("Expected score: {}".format(score))
import matplotlib.pyplot as plt
plt.figure(figsize=(30,20))
_ = plt.plot(city_coordinates[best_path, 0], city_coordinates[best_path, 1])
from collections import deque
def cost_around_point(index, point, before, after):
    ret  = np.linalg.norm(city_coordinates[point] - city_coordinates[before]) * (1 + (mask_tens[index - 1] * not_primes[before]))
    ret += np.linalg.norm(city_coordinates[point] - city_coordinates[after] ) * (1 + (mask_tens[index]     * not_primes[point] ))
    return ret

def simulated_annealing(curr_path, score):
    new_score = score
    new_path = list(curr_path)
    T0 = 2.0
    alpha = 0.9
    T_steps = 90
    Tvals = [T0 * alpha**x for x in range(T_steps)]
    np.random.seed(666)
    score_history = {'T':[],'avg_dists':[], 'anneals':[]}
    increase_counter = 0
    tried_swaps = deque(maxlen=100000)
    deltas = deque([3, 3,100, 100])
    for T in tqdm_notebook(Tvals):
        delta = deltas.popleft()
        annealing_steps = 0
        scores = []
        for _ in range(100000):
            while True:
                i1 = np.random.randint(1, num_cities)
                i2 = max(1, (i1 + np.random.poisson(delta)) % num_cities)
                
                if (i1,i2) not in tried_swaps and np.abs(i1-i2) >= 3:
                    break

            prev_i1i2  = cost_around_point(i1, new_path[i1], new_path[i1 - 1], new_path[i1 + 1]) 
            prev_i1i2 += cost_around_point(i2, new_path[i2], new_path[i2 - 1], new_path[i2 + 1])

            new_i1i2   = cost_around_point(i1, new_path[i2], new_path[i1 - 1], new_path[i1 + 1]) 
            new_i1i2  += cost_around_point(i2, new_path[i1], new_path[i2 - 1], new_path[i2 + 1])

            tentative_score = new_score - prev_i1i2 + new_i1i2

            if (tentative_score < new_score) or (np.random.rand() < np.exp((new_score - tentative_score) / T )):
                scores.append(tentative_score)
                tried_swaps.append((i1,i2))
                if tentative_score > new_score:
                    annealing_steps += 1
                new_path[i1], new_path[i2] = new_path[i2], new_path[i1]
                new_score = tentative_score
        if len(scores) > 0:
            score_history['T'].append(T)
            score_history['avg_dists'].append(np.average(scores))
            score_history['anneals'].append(annealing_steps)
        deltas.append(delta)
    return new_path, new_score, pd.DataFrame(score_history)
new_best_path, score, score_history = simulated_annealing(best_path, score)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
_ = score_history.plot(x='T', y='avg_dists',ax=axes[0])
_ = score_history.plot(x='T', y='anneals',ax=axes[1])
axes[0].invert_xaxis()
axes[1].invert_xaxis()
plt.figure(figsize=(80,30))
plt.subplot(1,2,1)
plt.plot(city_coordinates[best_path, 0], city_coordinates[best_path, 1])
_ = plt.title("Original solution", fontsize=40)
plt.subplot(1,2,2)
plt.plot(city_coordinates[new_best_path, 0], city_coordinates[best_path, 1])
_ = plt.title("Improved solution", fontsize=40)
submission = pd.DataFrame({"Path": new_best_path})
submission.to_csv("submission.csv", index=None)