import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
from collections import Counter
from time import time
from matplotlib import collections  as mc

sns.set_style('whitegrid')
df = pd.read_csv("../input/cities.csv")
df.head()
plt.figure(figsize=(15, 10))
plt.scatter(df.X, df.Y, s=1)
plt.scatter(df.iloc[0: 1, 1], df.iloc[0: 1, 2], s=10, c="red")
plt.grid(False)
plt.show()
nb_cities = max(df.CityId)
print("Number of cities to visit : ", nb_cities)
df.tail()
def sieve_eratosthenes(n):
    primes = [False, False] + [True for i in range(n-1)]
    p = 2
    while (p * p <= n):
        if (primes[p] == True):
            for i in range(p * 2, n + 1, p):
                primes[i] = False
        p += 1
    return primes
primes = np.array(sieve_eratosthenes(nb_cities)).astype(int)
df['Prime'] = primes
penalization = 0.1 * (1 - primes) + 1
df.head()
plt.figure(figsize=(15, 10))
sns.countplot(df.Prime)
plt.title("Prime repartition : " + str(Counter(df.Prime)))
plt.show()
plt.figure(figsize=(15, 10))
plt.scatter(df[df['Prime'] == 0].X, df[df['Prime'] == 0].Y, s=1, alpha=0.4)
plt.scatter(df[df['Prime'] == 1].X, df[df['Prime'] == 1].Y, s=1, alpha=0.6, c='blue')
plt.scatter(df.iloc[0: 1, 1], df.iloc[0: 1, 2], s=10, c="red")
plt.grid(False)
plt.title('Visualisation of cities')
plt.show()
def dist_matrix(coords, i, penalize=False):
    begin = np.array([df.X[i], df.Y[i]])[:, np.newaxis]
    mat =  coords - begin
    if penalize:
        return np.linalg.norm(mat, ord=2, axis=0) * penalization
    else:
        return np.linalg.norm(mat, ord=2, axis=0)
def get_next_city(dist, avail):
    return avail[np.argmin(dist[avail])]
coordinates = np.array([df.X, df.Y])
current_city = 0 
left_cities = np.array(df.CityId)[1:]
path = [0]
stepNumber = 1
t0 = time()

while left_cities.size > 0:
    if stepNumber % 10000 == 0: #We print the progress of the algorithm
        print(f"Time elapsed : {time() - t0} - Number of cities left : {left_cities.size}")
    # If we are at the ninth iteration (modulo 10), we may want to go to a prime city. Note that there is an approximation here: we penalize the path to the 10th city insted of 11th
    favorize_prime = (stepNumber % 10 == 9)
    # Compute the distance matrix
    distances = dist_matrix(coordinates, current_city, penalize=favorize_prime)
    # Get the closest city and go to it
    current_city = get_next_city(distances, left_cities)
    # Update the list of not visited cities
    left_cities = np.setdiff1d(left_cities, np.array([current_city]))
    # Append the city to the path
    path.append(current_city)
    # Add one step
    stepNumber += 1
print(f"Loop lasted {(time() - t0) // 60} minutes ")
path.append(0)
print(len(path) == len(df) + 1)
def plot_path(path, coordinates):
    # Plot tour
    lines = [[coordinates[: ,path[i-1]], coordinates[:, path[i]]] for i in range(1, len(path)-1)]
    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = plt.subplots(figsize=(20,20))
    ax.set_aspect('equal')
    plt.grid(False)
    ax.add_collection(lc)
    ax.autoscale()
plot_path(path, coordinates)
def get_score(path, coords, primes):
    score = 0
    for i in range(1, len(path)):
        begin = path[i-1]
        end = path[i]
        distance = np.linalg.norm(coords[:, end] - coords[:, begin], ord=2)
        if i%10 == 0:
            if not primes[begin]:
                distance *= 1.1
        score += distance
    return score
score = get_score(path, coordinates, primes)
print("Solution scored ", score)
print(score - get_score(path[:-1], coordinates, primes))
submission = pd.DataFrame({"Path": path})
submission.to_csv("submission.csv", index=None)