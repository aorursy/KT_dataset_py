import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

%matplotlib inline
df = pd.read_csv('../input/cities.csv', header=0)
df.info()
df.head()
print('X Range: [{min}, {max}]'.format(min=round(df['X'].min(), 2), max=round(df['X'].max(), 2)))
print('Y Range: [{min}, {max}]'.format(min=round(df['Y'].min(), 2), max=round(df['Y'].max(), 2)))
df.plot(x='X', y='Y', kind='scatter', alpha=0.005, color='Black')
def find_primes(n):
    if n < 2:
        return([])
    primes = [2]
    if n < 3:
        return(primes)
    
    for i in range(3, n+1):
        is_prime = True
        for p in primes:
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    return(primes)
def find_primes_in_list(l):
    primes_hashmap = {x: '' for x in find_primes(max(l))}
    return([x for x in l if x in primes_hashmap])
def compute_distance(pos_1, pos_2):
    if len(pos_1) != 2 or len(pos_2) != 2:
        return(None)
    return(np.sqrt((pos_1[0] - pos_2[0])**2 + (pos_1[1] - pos_2[1])**2))
def is_route_valid(path, city_ids=list(df['CityId'])):
    if len(path) != len(city_ids) + 1:
        return(False)
    if path[0] != 0 or path[len(city_ids)] != 0:
        return(False)
    if set(path) != set(city_ids):
        return(False)
    return(True)
def compute_route_distance(path_distance_dict):
    total_distance = 0
    city_ids = list(path_distance_dict)
    primes_hashmap = {x: '' for x in find_primes_in_list(city_ids)}
    for i in range(1, len(city_ids)):
        distance = compute_distance(path_distance_dict[city_ids[i-1]], path_distance_dict[city_ids[i]])
        if i % 10 == 0:
            if city_ids[i-1] not in primes_hashmap:
                distance *= 1.1
        total_distance += distance
    return(total_distance)
primes = find_primes_in_list(list(df['CityId']))
primes_hashmap = {x: '' for x in primes}
print('Number of cities: {count}'.format(count=len(df['CityId'])))
print('Number of prime cities: {count}'.format(count=len(primes)))
primes[:20]
path = list(range(0, 197769))
path.append(0)
is_route_valid(path)
path_distance_dict = {}
for row in df.values.tolist():
    path_distance_dict[int(row[0])] = row[1:]
path_distance_dict[len(path_distance_dict)] = path_distance_dict[0]

print(len(path_distance_dict))
for city_id in list(path_distance_dict)[:5]:
    pos = path_distance_dict[city_id]
    print('City ID: {city_id} had position: ({x}, {y})'.format(city_id=city_id, x=pos[0], y=pos[1]))
start_time = time.time()

total_distance = compute_route_distance(path_distance_dict)

print('Computing the route total distance took {time} seconds'.format(time=round(time.time() - start_time, 1)))

total_distance
