import numpy as np
import pandas as pd
import numba
from sympy import isprime, primerange
from math import sqrt
from sklearn.neighbors import KDTree
from tqdm import tqdm_notebook as tqdm
from itertools import combinations, permutations
from functools import lru_cache
cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv', index_col=['CityId'])
XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)
@numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
def cities_distance(offset, id_from, id_to):
    xy_from, xy_to = XY[id_from], XY[id_to]
    dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
    distance = sqrt(dx * dx + dy * dy)
    if offset % 10 == 9 and is_not_prime[id_from]:
        return 1.1 * distance
    return distance


@numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
def score_chunk(offset, chunk):
    pure_distance, penalty = 0.0, 0.0
    penalty_modulo = 9 - offset % 10
    for path_index in numba.prange(chunk.shape[0] - 1):
        id_from, id_to = chunk[path_index], chunk[path_index+1]
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        pure_distance += distance
        if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
            penalty += distance
    return pure_distance + 0.1 * penalty


@numba.jit('f8(i8[:])', nopython=True, parallel=False)
def score_path(path):
    return score_chunk(0, path)


@numba.jit
def chunk_scores(chunk):
    scores = np.zeros(10)
    pure_distance = 0
    for i in numba.prange(chunk.shape[0] - 1):
        id_from, id_to = chunk[i], chunk[i+1]
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        pure_distance += distance
        if is_not_prime[id_from]:
            scores[9-i%10] += distance
    scores *= 0.1
    scores += pure_distance
    return scores
@numba.jit('f8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:])', nopython=True, parallel=False)
def score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes):
    score = 0.0
    last_city_id = head
    for i in numba.prange(len(indexes)):
        index = indexes[i]
        first, last, chunk_len = firsts[index], lasts[index], lens[index]
        score += cities_distance(offset, last_city_id, first)
        score += scores[index, (offset + 1) % 10]
        last_city_id = last
        offset += chunk_len
    return score + cities_distance(offset, last_city_id, tail)


@numba.jit('i8(i8, i8, i8[:], i8[:], i8[:], i8, f8[:,:], i8[:,:], f8)', nopython=True, parallel=False)
def best_score_permutation_index(offset, head, firsts, lasts, lens, tail, scores, indexes, best_score):
    best_index = -1
    for i in numba.prange(len(indexes)):
        score = score_compound_chunk(offset, head, firsts, lasts, lens, tail, scores, indexes[i])
        if score < best_score:
            best_index, best_score = i, score
    return best_index
kdt = KDTree(XY)

fives = set()
for i in tqdm(cities.index):
    dists, neibs = kdt.query([XY[i]], 9)
    for comb in combinations(neibs[0], 5):
        if all(comb):
            fives.add(tuple(sorted(comb)))
    neibs = kdt.query_radius([XY[i]], 10, count_only=False, return_distance=False)
    for comb in combinations(neibs[0], 5):
        if all(comb):
            fives.add(tuple(sorted(comb)))
            
print(f'{len(fives)} cities fives are selected.')

# sort fives by distance
@numba.jit('f8(i8[:])', nopython=True, parallel=False)
def sum_distance(ids):
    res = 0
    for i in numba.prange(len(ids)):
        for j in numba.prange(i + 1, len(ids)):
            res += cities_distance(0, ids[i], ids[j])
    return res

fives = np.array(list(fives))
distances = np.array(list(map(sum_distance, tqdm(fives))))
order = distances.argsort()
fives = fives[order]
path = np.array(pd.read_csv('../input/shame-on-me/1515656.5970491648.csv').Path)
@lru_cache(maxsize=None)
def indexes_permutations(n):
    return np.array(list(map(list, permutations(range(n)))))


path_index = np.argsort(path[:-1])
print(f'Total score is {score_path(path):.2f}.')
for _ in range(2):
    for ids in tqdm(fives[:2 * 10**6]):
        i1, i2, i3, i4, i5 = np.sort(path_index[ids])
        head, tail = path[i1-1], path[i5+1]
        chunks = [path[i1:i1+1], path[i1+1:i2], path[i2:i2+1], path[i2+1:i3],
                  path[i3:i3+1], path[i3+1:i4], path[i4:i4+1], path[i4+1:i5], path[i5:i5+1]]
        chunks = [chunk for chunk in chunks if len(chunk)]
        scores = np.array([chunk_scores(chunk) for chunk in chunks])
        lens = np.array([len(chunk) for chunk in chunks])
        firsts = np.array([chunk[0] for chunk in chunks])
        lasts = np.array([chunk[-1] for chunk in chunks])
        best_score = score_compound_chunk(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks))[0])
        index = best_score_permutation_index(i1-1, head, firsts, lasts, lens, tail, scores, indexes_permutations(len(chunks)), best_score)
        if index > 0:
            perm = [chunks[i] for i in indexes_permutations(len(chunks))[index]]
            path[i1-1:i5+2] = np.concatenate([[head], np.concatenate(perm), [tail]])
            path_index = np.argsort(path[:-1])
            print(f'New total score is {score_path(path):.3f}. Permutating path at indexes {i1}, {i2}, {i3}, {i4}, {i5}.')
def make_submission(name, path):
    pd.DataFrame({'Path': path}).to_csv(f'{name}.csv', index=False)


make_submission(score_path(path), path)