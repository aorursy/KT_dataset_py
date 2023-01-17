import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os
print(os.listdir("../input"))

import numpy as np
import time

df = pd.read_csv("../input/cities.csv")
df.info()
def is_prime(number):
    prime = False
    if number == 2:
        prime = True
    elif number > 1:
    # check for factors
        for i in range(2,np.ceil(np.sqrt(number)).astype(int) + 1):
            if (number % i) == 0:
                break
        else:
            prime = True
    return prime

df['prime'] = df['CityId'].apply(is_prime)
def euclidean_distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)
def score(list_of_cities, df, start_at=0, queue=None):
    total_distance = 0
    for counter, CityId in enumerate(list_of_cities[:-1]):
        origin_city_coords = df[df['CityId'] == CityId][['X', 'Y']].iloc[0]
        dest_city_coords = df[df['CityId'] == list_of_cities[counter + 1]][['X', 'Y']].iloc[0]
        distance = euclidean_distance(origin_city_coords['X'], origin_city_coords['Y'], dest_city_coords['X'],
                                      dest_city_coords['Y'])
        if (start_at + counter + 1) % 10 == 0 and not df[df['CityId'] == CityId]['prime'].iloc[0]:
            distance *= 1.1
        total_distance += distance

    if queue is not None:
        queue.put(total_distance)

    return total_distance
def parallel_score(list_of_cities, df, cores=4):
    queue = multiprocessing.Queue()
    step = len(list_of_cities) // cores
    jobs = []
    start_time = time.time()
    
    
    #The cities at the end of one list and the beginning of the next won't be processing, so we need to do 'manually' 
    union_list_cities = []
    # NOTE: We are no dealing with the step number here cause the number of the cities divided by the kernels cores (4) is not divisible by 10, so no constrain is applied.
    
    for x in np.arange(0, cores):
        if x == cores - 1:
            sublist = list_of_cities[step * x:]
        else:
            sublist = list_of_cities[step * x: step * (x + 1)]
            
        if sublist[0] != 0:
            union_list_cities.append(sublist[0])
        if sublist[-1] != 0:
            union_list_cities.append(sublist[-1])
        
        p = multiprocessing.Process(target=score, args=(sublist, df, step * x, queue))
        p.start()
        jobs.append(p)

    for job in jobs:
        job.join()
    
    
    print("Total computation time: {} with {} cores".format(time.time() - start_time, cores))
    return sum([queue.get() for _ in np.arange(0, cores)]) + sum([score(x,df) for x in zip(union_list_cities[::2], union_list_cities[1::2])])
#Using 1 core. Standar score
parallel_score(df['CityId'].tolist() + [0], df, cores=1)
#Full power
parallel_score(df['CityId'].tolist() + [0], df)