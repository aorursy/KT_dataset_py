#Using Content from and probably others too
#https://www.kaggle.com/blacksix/dp-shuffle
#https://www.kaggle.com/matthewa313/flip-it
#https://www.kaggle.com/byfone/riffling-for-fine-selection

import numpy as np
import pandas as pd
from sympy import primerange
from itertools import permutations
from multiprocessing import Pool, cpu_count

def score_tour(tour):
    df = cities.reindex(tour + [0]).reset_index()
    df['prime'] = df.CityId.isin(primes).astype(int)
    df['dist'] = np.hypot(df.X - df.X.shift(-1), df.Y - df.Y.shift(-1))
    df['penalty'] = df['dist'][9::10] * (1 - df['prime'][9::10]) * 0.1
    return df.dist.sum() + df.penalty.sum()

cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv')
primes = list(primerange(0, len(cities['CityId'].values)))
primes += [0,1]
cities['NotPrime'] = np.abs(cities.CityId.isin(primes).astype(int) - 1) * 0.1 + 1
dPaths = {v:{'X': x, 'Y': y, 'penalty': z} for v, x, y, z in cities[['CityId','X','Y','NotPrime']].values}
cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv', index_col=['CityId'])
tour = pd.read_csv('../input/dp-shuffle/submission.csv')['Path'].tolist()
score_tour(tour)
def bscore(cand, is_tenth):
    s = 0.0
    for j in range(len(cand)-1):
        p = np.hypot(dPaths[cand[j]]['X'] - dPaths[cand[j+1]]['X'], dPaths[cand[j]]['Y'] - dPaths[cand[j+1]]['Y'])
        if is_tenth[j]:
            p *= dPaths[j]['penalty']
        s += p
    return s

def riffle2(batch, i):
    is_tenth = [(j+1)%10==0 for j in range(i,i+n)]
    best = batch
    for per in permutations(batch[1:-1]):
        perm = [batch[0]]+list(per)+[batch[-1]]
        if bscore(perm, is_tenth) < bscore(best, is_tenth):
            best = perm
    if best != batch:
        return best
    else:
        return None

def riffle(split_tour1):
    order, batch = split_tour1
    for i in range(0, len(batch)-n+1):
        r = riffle2(batch[i:i+n],i)
        if r:
            print(r,i)
            batch = batch[:i] + r + batch[i+n:]
    return [order, batch]

def multi_riffle(t):
    ret_d = {}
    p = Pool(cpu_count())
    t = np.array_split(t, 10000)
    t = [[i,list(t)] for i, t in enumerate(t)]
    ret = p.map(riffle, t)
    for i in range(len(ret)):
        ret_d[ret[i][0]] = ret[i][1]
    ret = []
    for i in range(len(ret_d)):
        ret += ret_d[i]
    return ret
%%time
n=8
tour = multi_riffle(tour)
pd.DataFrame({'Path': list(tour)}).to_csv('submission.csv', index=False)
print(score_tour(tour))