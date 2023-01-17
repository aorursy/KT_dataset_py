import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pickle
import os
import itertools
from scipy.sparse import csr_matrix
import os

import os
print(os.listdir("../input/reversing-and-shifting"))
cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv')

cities_numb = len(cities)
subm = pd.read_csv('../input/reversing-and-shifting/submission.csv')
list_submits = [x for x in os.listdir("../input/") if x not in ['traveling-santa-2018-prime-paths', 'reversing-and-shifting']]
filenames = []
for s in list_submits:
    for ss in os.listdir("../input/"+ s):
        if ss[-4:] == '.csv':
            check = pd.read_csv("../input/"+ s + '/' +  ss)
            if 'Path' in check.columns:
                filenames += ["../input/"+ s + '/' +  ss]
path_dict = {}
sub_perm_dict = {}

for file_name in filenames:
    path_dict[file_name] = pd.read_csv(file_name).Path.tolist()
    sub_perm_dict[file_name] = np.zeros(len(path_dict[file_name]))
    for j, x in enumerate(path_dict[file_name][:-1]):
        sub_perm_dict[file_name][x] = j
def get_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
city_X = np.zeros(cities_numb)
city_Y = np.zeros(cities_numb)

for city, x, y in zip(cities.CityId.tolist(), cities.X.tolist(), cities.Y.tolist()):
    city_X[city] = x
    city_Y[city] = y
def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'))


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))
def SieveOfEratosthenes(n):
    # Create a boolean array "prime[0..n]" and initialize
    #  all entries it as true. A value in prime[i] will
    # finally be false if i is Not a prime, else true.
    prime = [True for i in range(n + 1)]
    prime[0] = False
    prime[1] = False
    p = 2
    while (p * p <= n):

        # If prime[p] is not changed, then it is a prime
        if (prime[p] == True):

            # Update all multiples of p
            for i in range(p * 2, n + 1, p):
                prime[i] = False
        p += 1
    return prime
def get_primes():
    cache_path = 'prime_list.pkl'
    if not os.path.isfile(cache_path):
        n = 200000
        prime = SieveOfEratosthenes(n)
        plist = []
        for p in range(2, n):
            if prime[p]:
                plist.append(p)
        save_in_file_fast(set(plist), cache_path)
    else:
        plist = load_from_file_fast(cache_path)

    return plist
plist = get_primes()
def get_score(santa_path, numb_list):
    sum_dist = 0
    for i in range(len(santa_path) - 1):
        city1 = santa_path[i]
        city2 = santa_path[i+1]
        if city1 > city2:
            city1, city2 = city2, city1
        x1 = city_X[city1]
        y1 = city_Y[city1]
        x2 = city_X[city2]
        y2 = city_Y[city2]
        
        city1_dict = dist_dict[city1]
        
        if city2 in city1_dict:
            dist = city1_dict[city2]
        else:
            dist = get_dist(x1, y1, x2, y2)
            dist_dict[city1][city2] = dist
        if ((numb_list[i] + 1) % 10 == 0) and (not santa_path[i] in plist):
            dist *= 1.1
        sum_dist += dist
    return sum_dist
best_path = subm['Path'].tolist()
dist_dict = {i:{} for i in range(cities_numb)}
def pop(cur_perm, numb_list, score):
    city1 = cur_perm[0]
    city2 = cur_perm[1]
    x1 = city_X[city1]
    y1 = city_Y[city1]
    x2 = city_X[city2]
    y2 = city_Y[city2]
    dist = get_dist(x1, y1, x2, y2)
    pos = numb_list[0]
    
    if ((pos + 1) % 10 == 0 and (not city1 in plist)):
        return cur_perm[1:], numb_list[1:], score - dist * 1.1
    else:
        return cur_perm[1:], numb_list[1:], score - dist
    
def push(cur_perm, numb_list, score, x):
    city1 = cur_perm[-1]
    city2 = x
    x1 = city_X[city1]
    y1 = city_Y[city1]
    x2 = city_X[city2]
    y2 = city_Y[city2]
    dist = get_dist(x1, y1, x2, y2)
    pos = numb_list[-1] + 1
    
    if ((pos) % 10 == 0 and (not city1 in plist)):
        return cur_perm + [x], numb_list + [pos], score + dist * 1.1
    else:
        return cur_perm + [x], numb_list + [pos], score + dist
def improve_perm(santa_path, len_perm):
    santa_len = len(santa_path)
    L = 150000
    R = len_perm + L
    numb_list = list(range(L, R))
    cur_perm = santa_path[L:R]
    cur_dist = get_score(cur_perm, numb_list)
    
    for i in tqdm(range(0, santa_len - R - 1)):
        if i % 15 == 0:
            for file_name in filenames:
                new_path = []
                indx = sorted([sub_perm_dict[file_name][x] for x in cur_perm[1:-1]])
                for ind in indx:
                    new_path += [path_dict[file_name][int(ind)]]

                cur_path = [cur_perm[0]] + new_path + [cur_perm[-1]]
                cur_path_rev = [cur_perm[0]] + new_path[::-1] + [cur_perm[-1]]

                cur_score = get_score(cur_path, numb_list)
                if (cur_score < cur_dist and cur_dist - cur_score > 0.05):
                    print('GO ', cur_score, cur_dist, get_score(cur_perm, numb_list))
                    ans = list(santa_path)
                    ans[L:R] = cur_path
                    return ans
                cur_score = get_score(cur_path_rev, numb_list)
                if (cur_score < cur_dist and cur_dist - cur_score > 0.05):
                    print('REV ', cur_score, cur_dist)
                    ans = list(santa_path)
                    ans[L:R] = cur_path_rev
                    return ans
        cur_perm, numb_list, cur_dist = pop(cur_perm, numb_list, cur_dist)
        cur_perm, numb_list, cur_dist = push(cur_perm, numb_list, cur_dist, santa_path[R])
        L += 1
        R += 1
best_path2 = improve_perm(best_path, 100)
get_score(best_path, list(range(len(best_path))))
get_score(best_path2, list(range(len(best_path2))))
ans = pd.DataFrame()
ans['Path'] = best_path2
ans.to_csv('submission.csv', index = None)
