import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sympy import primerange
from itertools import permutations
def score_tour(tour):
    df = cities.reindex(tour + [0]).reset_index()
    df['prime'] = df.CityId.isin(primes).astype(int)
    df['dist'] = np.hypot(df.X - df.X.shift(-1), df.Y - df.Y.shift(-1))
    df['penalty'] = df['dist'][9::10] * (1 - df['prime'][9::10]) * 0.1
    return df.dist.sum() + df.penalty.sum()
cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv', index_col=['CityId'])
primes = list(primerange(0, len(cities)))
tour = pd.read_csv('../input/concorde-for-5-hours/submission.csv')['Path'].tolist()
score_tour(tour)
n = 7

def bscore(cand, is_prime, is_tenth):
    s = 0.0
    for j in range(len(cand)-1):
        p = np.hypot(cities.loc[cand[j], 'X'] -
                 cities.loc[cand[j+1], 'X'],
                 cities.loc[cand[j], 'Y'] -
                 cities.loc[cand[j+1], 'Y'])
        if is_tenth[j] and not is_prime[cand[j]]:
            p = p * 1.1
        s += p
    return s

def riffle(batch,i):
    is_prime = {c: c in primes for c in batch}
    is_tenth = [(j+1)%10==0 for j in range(i,i+n)]
    best = batch
    for per in permutations(batch[1:-1]):
        perm = [batch[0]]+list(per)+[batch[-1]]
        if bscore(perm, is_prime, is_tenth) < bscore(best, is_prime, is_tenth):
            best = perm
    if best != batch:
        return best
    else:
        return None

for i in range(0, len(tour)-n+1):
    if i%10000==0:
        print(i)
    r = riffle(tour[i:i+n],i)
    if r:
        print(r,i)
        tour = tour[:i] + r + tour[i+n:]
pd.DataFrame({'Path': list(tour)}).to_csv('submission.csv', index=False)
score_tour(tour)