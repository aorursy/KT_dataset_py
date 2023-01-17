import numpy as np

from scipy.signal import convolve2d

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from pprint import pprint



window = np.ones((3, 3))



def plot_3d(solution_3d: np.ndarray, title, size=3, max_cols=10, has_target=False):

    N = len(solution_3d)

    if N <= 0:

        return

    cols = min(N, max_cols)

    rows = (N - 1) // max_cols + 1

    plt.figure(figsize=(cols*size, rows*size))

    plt.suptitle(title)

    for t in range(N):

        board = solution_3d[t]

        plt.subplot(rows, cols, t + 1)

        plt.imshow(board, cmap='binary')

        plt.title('target' if t == 0 and has_target else f'source {t}')

    plt.show()



def life_step(X: np.ndarray):

    C = convolve2d(X, window, mode='same', boundary='wrap')

    return (C == 3) | (X & (C == 4))



SIZE = 4



empty_board = np.zeros((SIZE,SIZE), dtype=bool)

m2i = np.array([2**i for i in range(SIZE**2)]).reshape(SIZE,SIZE)



def is_d_step_child(n0, d, children):

    n1 = n0

    while d > 0:

        n1 = children[n1]

        d -= 1

        if n1 == n0:

            return d == 0

    return n1 == n0



r = np.array([2**i for i in range(SIZE)], dtype=int)

rr = np.cumsum(np.array([2**(i-1) for i in range(SIZE)], dtype=int))

c = np.array([2**(i*SIZE) for i in range(SIZE)], dtype=int)

cc = np.cumsum(np.array([2**((i-1)*SIZE) for i in range(SIZE)], dtype=int))

sr = np.sum(r)

sc = np.sum(c)

mr = 2**SIZE

mc = 2**(SIZE*SIZE)



def h_flip(x):

    x = ((x & 0x5555) << 1) | ((x & 0xAAAA) >> 1)

    x = ((x & 0x3333) << 2) | ((x & 0xCCCC) >> 2)

    return x

    

def v_flip(x):

    x = ((x & 0x0F0F) << 4) | ((x & 0xF0F0) >> 4)

    x = ((x & 0x00FF) << 8) | ((x & 0xFF00) >> 8)

    return x



def d_flip(x):

    x = ((x & 0x0001) << 15) | ((x & 0x0012) << 10) | ((x & 0x0124) << 5) | (x & 0x1248) | ((x & 0x2480) >> 5) | ((x & 0x4800) >> 10) | ((x & 0x8000) >> 15) 

    return x



def hash(n):

    dn = d_flip(n)

    hn = h_flip(n)

    hdn = h_flip(dn)

    n = np.array([n, v_flip(n), hn, v_flip(hn), dn, v_flip(dn), hdn, v_flip(hdn)]).reshape((1,-1,1))

    n = (n & (cc * sr)) * mc // c + (n & ((sc - cc) * sr)) // c

    n = n.T

    n = (n & (rr * sc)) * mr // r + (n & ((sr - rr) * sc)) // r

    return np.min(n)
N = 2**(SIZE**2)

board = np.tile(empty_board, (2, 1, 1))

children = np.zeros(N, dtype=int)

mapping = {}

for n0 in tqdm(range(N), total=N):

    board[0] = (m2i & n0) != 0

    board[1] = life_step(board[0])

    n1 = np.sum(board[1] * m2i)

    children[n0] = n1

    n1 = hash(n1)

    if n1 not in mapping:

        mapping[n1] = set()

    mapping[n1].add(hash(n0))
static = list(set([hash(n0) for n0 in range(N) if is_d_step_child(n0, 1, children)]))

S = len(static)

board = np.tile(empty_board, (S, 1, 1))

for s in tqdm(range(S), total=S):

    board[s] = (m2i & static[s]) != 0

plot_3d(board, f'Static boards')
D = 16

cyclic = [[]]*D

for d in tqdm(range(D), total=D):

    cyclic[d] = list(set([hash(n0) for n0 in range(N) if is_d_step_child(n0, d+1, children)]))

print({d+1:len(cyclic[d]) for d in range(D)})
for d in tqdm(range(D), total=D):

    S = len(cyclic[d])

    board = np.tile(empty_board, (S, 1, 1))

    for s in tqdm(range(S), total=S):

        board[s] = (m2i & cyclic[d][s]) != 0

    plot_3d(board, f'{d+1}-step cyclic boards')
mapping = {k:mapping[k] for k in mapping if all(k not in cyclic[d] for d in range(D))}

pprint({n1:len(mapping[n1]) for n1 in mapping})
depth = {n1:0 for n1 in range(N)}

changed = True

while changed:

    changed = False

    for n1 in range(N):

        if n1 in mapping:

            for n0 in mapping[n1]:

                if n0 != n1 and depth[n1] < depth[n0] + 1:

                    depth[n1] = depth[n0] + 1

                    changed = True
for n1 in range(N):

    if n1 in mapping:

        max_depth = max(depth[n0] for n0 in mapping[n1])

        mapping[n1] = [n0 for n0 in mapping[n1] if depth[n0] == max_depth]
keys = sorted(mapping.keys())

M = len(keys)

for n1 in tqdm(keys, total=M):

    board = np.tile(empty_board, (len(mapping[n1])+1, 1, 1))

    board[0] = (m2i & n1) != 0

    board[1:] = [(m2i & n0) != 0 for n0 in mapping[n1]]

    plot_3d(board, f'{n1} (depth {depth[n1]})', has_target=True)