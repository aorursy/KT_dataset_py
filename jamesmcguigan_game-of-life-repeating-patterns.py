import numpy  as np

import pandas as pd

import math

import matplotlib.pyplot as plt

from itertools import chain, product

from scipy.signal import convolve2d

from numba import njit

from joblib import delayed

from joblib import Parallel

from typing import Union, List, Dict, Tuple, FrozenSet, Set

from collections import defaultdict
input_directory = f'../input/conways-reverse-game-of-life-2020/'

train_file      = f'{input_directory}/train.csv'

test_file       = f'{input_directory}/test.csv'



train_df        = pd.read_csv(train_file, index_col='id')

test_df         = pd.read_csv(test_file,  index_col='id')
# NOTE: @njit doesn't like np.roll(axis=) so reimplement explictly

@njit

def life_neighbours_xy(board: np.ndarray, x, y, max_value=3):

    size_x = board.shape[0]

    size_y = board.shape[1]

    neighbours = 0

    for i in (-1, 0, 1):

        for j in (-1, 0, 1):

            if i == j == 0: continue    # ignore self

            xi = (x + i) % size_x

            yj = (y + j) % size_y

            neighbours += board[xi, yj]

            if neighbours > max_value:  # shortcircuit return 4 if overpopulated

                return neighbours

    return neighbours



@njit

def life_step(board: np.ndarray):

    """Game of life step using generator expressions"""

    size_x = board.shape[0]

    size_y = board.shape[1]

    output = np.zeros(board.shape, dtype=np.int8)

    for x in range(size_x):

        for y in range(size_y):

            cell       = board[x,y]

            neighbours = life_neighbours_xy(board, x, y, max_value=3)

            if ( (cell == 0 and      neighbours == 3 )

              or (cell == 1 and 2 <= neighbours <= 3 )

            ):

                output[x, y] = 1

    return output



def csv_to_numpy(df, idx, key='start') -> np.ndarray:

    columns = [col for col in df if col.startswith(key)]

    size    = int(math.sqrt(len(columns)))

    board   = df.loc[idx][columns].values

    if len(board) == 0: board = np.zeros((size, size))

    board = board.reshape((size,size)).astype(np.int8)

    return board





def plot_3d(solution_3d: np.ndarray, size=4, max_cols=6):

    cols = np.min([ len(solution_3d), max_cols ])

    rows = len(solution_3d) // cols + 1

    plt.figure(figsize=(cols*size, rows*size))

    for t in range(len(solution_3d)):

        board = solution_3d[t]

        plt.subplot(rows, cols, t + 1)

        plt.imshow(board, cmap='binary'); plt.title(f't={t}')

    plt.show()





def crop_inner(grid,tol=0):

    mask = grid > tol

    return grid[np.ix_(mask.any(1),mask.any(0))]





def crop_outer(grid,tol=0):

    mask = grid>tol

    m,n  = grid.shape

    mask0,mask1 = mask.any(0), mask.any(1)

    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()

    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()

    return grid[row_start:row_end,col_start:col_end]





def crop_and_center(board: np.ndarray, shape=(25,25)) -> Union[np.ndarray, None]:

    cropped = crop_outer(board)

    offset  = ( (shape[0]-cropped.shape[0])//2, (shape[1]-cropped.shape[1])//2 ) 

    zeros   = np.zeros(shape, dtype=np.int8)

    zeros[ offset[0]:offset[0]+cropped.shape[0], offset[1]:offset[1]+cropped.shape[1] ] = cropped

    return zeros

    



def filter_crop_and_center(board: np.ndarray, max_size=6, shape=(25,25)) -> Union[np.ndarray, None]:

    for _ in range(2):

        cropped = crop_outer(board)

        if cropped.shape != crop_inner(cropped).shape: continue  # exclude multi-piece shapes

        if cropped.shape[0] <= max_size and cropped.shape[1] <= max_size:

            offset = ( (shape[0]-cropped.shape[0])//2, (shape[1]-cropped.shape[1])//2 ) 

            zeros  = np.zeros(shape)

            zeros[ offset[0]:offset[0]+cropped.shape[0], offset[1]:offset[1]+cropped.shape[1] ] = cropped

            return zeros

        else:

            # roll viewpoint and try again             

            board = np.roll(np.roll(board, shape[0]//2, axis=0), shape[1]//2, axis=1)               

    return None





def crop_outer_3d(solution_3d: np.ndarray, tol=0) -> np.ndarray:

    assert len(solution_3d.shape) == 3

    size_t,size_x,size_y = solution_3d.shape



    mask_t = np.array([ np.any(grid) for grid in solution_3d ])

    mask_x = np.any([ grid.any(axis=0) for grid in solution_3d ], axis=0)

    mask_y = np.any([ grid.any(axis=1) for grid in solution_3d ], axis=0)



    t_start,   t_end   = mask_t.argmax(), size_t - mask_t[::-1].argmax()

    col_start, col_end = mask_x.argmax(), size_x - mask_x[::-1].argmax()

    row_start, row_end = mask_y.argmax(), size_y - mask_y[::-1].argmax()

    output = solution_3d[ t_start:t_end, col_start:col_end, row_start:row_end ]

    return output





def crop_and_center_3d(solution_3d: np.ndarray, shape=(25,25)) -> Union[np.ndarray, None]:

    cropped = crop_outer_3d(solution_3d)

    offset  = ( (shape[0]-cropped[0].shape[0])//2, (shape[1]-cropped[0].shape[1])//2 )

    zeros   = np.zeros((cropped.shape[0], *shape), dtype=np.int8)

    zeros[ :, offset[0]:offset[0]+cropped[0].shape[0], offset[1]:offset[1]+cropped[0].shape[1] ] = cropped

    return zeros
import numpy as np

from numba import njit



# First 625 prime numbers

primes_np = np.array([ 

    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603, 4621, 4637

], dtype=np.int64)



# Hashable primes that don't have summation collisions for permutations=2 

hashable_primes = np.array([

        2,     7,    23,    47,    61,     83,    131,    163,    173,    251,

      457,   491,   683,   877,   971,   2069,   2239,   2927,   3209,   3529,

     4451,  4703,  6379,  8501,  9293,  10891,  11587,  13457,  13487,  17117,

    18869, 23531, 23899, 25673, 31387,  31469,  36251,  42853,  51797,  72797,

    76667, 83059, 87671, 95911, 99767, 100801, 100931, 100937, 100987, 100999,

], dtype=np.int64)





@njit()

def hash_geometric(board: np.ndarray) -> int:

    """

    Takes the 1D pixelwise view from each pixel (up, down, left, right) with wraparound

    the distance to each pixel is encoded as a prime number, the sum of these is the hash for each view direction

    the hash for each cell is the product of view directions and the hash of the board is the sum of these products

    this produces a geometric invariant hash that will be identical for roll / flip / rotate operations

    """

    assert board.shape[0] == board.shape[1]  # assumes square board

    size     = board.shape[0]

    l_primes = hashable_primes[:size//2+1]   # each distance index is represented by a different prime

    r_primes = l_primes[::-1]                # symmetric distance values in reversed direction from center



    hashed = 0

    for x in range(size):

        for y in range(size):

            # current pixel is moved to center [13] index

            horizontal = np.roll( board[:,y], size//2 - x)

            vertical   = np.roll( board[x,:], size//2 - y)

            left       = np.sum( horizontal[size//2:]   * l_primes )

            right      = np.sum( horizontal[:size//2+1] * r_primes )

            down       = np.sum( vertical[size//2:]     * l_primes )

            up         = np.sum( vertical[:size//2+1]   * r_primes )

            hashed    += left * right * down * up

    return hashed





@njit()

def hash_translations(board: np.ndarray) -> int:

    """

    Takes the 1D pixelwise view from each pixel (left, down) with wraparound

    by only using two directions, this hash is only invariant for roll operations, but not flip or rotate

    this allows determining which operations are required to solve a transform



    NOTE: np.rot180() produces the same sum as board, but with different numbers which is fixed via: sorted * primes

    """

    assert board.shape[0] == board.shape[1]

    hashes = hash_translations_board(board)

    sorted = np.sort(hashes.flatten())

    hashed = np.sum(sorted[::-1] * primes_np[:len(sorted)])  # multiply big with small numbers | hashable_primes is too small

    return int(hashed)





@njit()

def hash_translations_board(board: np.ndarray) -> np.ndarray:

    """ Returns a board with hash values for individual cells """

    assert board.shape[0] == board.shape[1]  # assumes square board

    size = board.shape[0]



    # NOTE: using the same list of primes for each direction, results in the following identity splits:

    # NOTE: np.rot180() produces the same np.sum() hash, but using different numbers which is fixed via: sorted * primes

    #   with v_primes == h_primes and NOT sorted * primes:

    #       identity == np.roll(axis=0) == np.roll(axis=1) == np.rot180()

    #       np.flip(axis=0) == np.flip(axis=1) == np.rot90() == np.rot270() != np.rot180()

    #   with v_primes == h_primes and sorted * primes:

    #       identity == np.roll(axis=0) == np.roll(axis=1)

    #       np.flip(axis=0) == np.rot270()

    #       np.flip(axis=1) == np.rot90()

    h_primes = hashable_primes[ 0*size : 1*size ]

    v_primes = hashable_primes[ 1*size : 2*size ]

    output   = np.zeros(board.shape, dtype=np.int64)

    for x in range(size):

        for y in range(size):

            # current pixel is moved to left [0] index

            horizontal  = np.roll( board[:,y], -x )

            vertical    = np.roll( board[x,:], -y )

            left        = np.sum( horizontal * h_primes )

            down        = np.sum( vertical   * v_primes )

            output[x,y] = left * down

    return output
def find_repeating_patterns(start_board: np.ndarray, delta=16, geometric=False) -> Union[np.ndarray, None]:

    """ Take 10 steps forward and check to see if the same pattern repeats """

    def hash_fn(board):

        return hash_geometric(board) if geometric else hash_translations(board)

    def is_symmetric(board): 

        return np.all( board == np.flip(board, axis=0) ) or np.all( board == np.flip(board, axis=0) )

    

    solution_3d = [ start_board ]

    hashes      = [ hash_fn(start_board) ]

    symmetric   = is_symmetric(start_board)

    for t in range(delta):

        next_board = life_step(solution_3d[-1])

        next_hash  = hash_fn(next_board)

        symmetric  = symmetric and is_symmetric(next_board)

        solution_3d.append(next_board)

        hashes.append( hash_fn(next_board) )

        if np.count_nonzero(next_board) == 0: break  # ignore dead boards

        if next_hash in hashes[:-1]:

            return crop_and_center_3d(np.array(solution_3d))

        

    if symmetric and len(solution_3d) > 5:

        return crop_and_center_3d(np.array(solution_3d))

    return None
def dataset_patterns() -> List[np.ndarray]:

    boards = [

        *[ csv_to_numpy(train_df, idx, key='start') for idx in train_df.index ],

        *[ csv_to_numpy(train_df, idx, key='stop')  for idx in train_df.index ],

        *[ csv_to_numpy(test_df,  idx, key='stop')  for idx in test_df.index  ],

    ]

    boards = [ filter_crop_and_center(board, max_size=6, shape=(25,25)) for board in boards ]

    boards = [ board for board in boards if board is not None ]

    hashes = Parallel(-1)([ delayed(hash_geometric)(board) for board in boards ])

    boards = list({ hashed: board for hashed, board in zip(hashes, boards) }.values())  # deduplicate

    patterns = Parallel(-1)([ delayed(find_repeating_patterns)(board, delta=16, geometric=False) for board in boards ])

    patterns = [ pattern for pattern in patterns if pattern is not None ]

    return patterns







@njit

def tail_repeating_pattern(pattern):

    """ Here we just want to show the actual repeating part """

    for t in range(len(pattern)):

        if np.all( pattern[t] == pattern[-1] ):

            return pattern[t:]



        

def tail_repeating_patterns(patterns):

    tails  = Parallel(-1)( delayed(tail_repeating_pattern)(pattern) for pattern in patterns )

    output = list({

        pattern.tobytes(): pattern

        for pattern in tails

        if len(pattern) > 1

    }.values())  # make unique

    return output
patterns       = dataset_patterns()

patterns_tails = tail_repeating_patterns(patterns)
for solution_3d in patterns_tails: plot_3d(solution_3d)
for solution_3d in patterns: plot_3d(solution_3d)
%%time



def generate_boards(shape=(4,4)):

    """Generate all possible board combinations of 1s and 0s, geometrically deduplicated"""

    sequences = product(range(2), repeat=np.product(shape)) 

    boards    = ( np.array(list(sequence), dtype=np.int8).reshape(shape) for sequence in sequences )

    boards    = ( crop_and_center(board) for board in boards )

    unique    = { board.tobytes(): board for board in boards }

    unique    = { hash_geometric(board): board for board in unique.values() }

    output = unique.values()

    return list(output)

  

    

def generated_patterns(shape=(4,4)) -> List[np.ndarray]:

    """All possible 3D boards with repeating/static patterns """

    boards   = generate_boards(shape=shape)

    boards   = list({ board.tobytes(): board for board in boards }.values())

    patterns = Parallel(-1)([ delayed(find_repeating_patterns)(board, delta=16, geometric=False) for board in boards ])

    patterns = [ pattern for pattern in patterns if pattern is not None ]

    return patterns





def grouped_patterns(patterns: List[np.ndarray]) -> Dict[bytes, np.ndarray]:

    """Group patterns by their final state"""

    

    # Group by 3D geometric hash

    index = {}

    for pattern_3d in patterns:

        t0_key = np.product([ hash_geometric(board) for board in pattern_3d[:] ])

        index[t0_key] = pattern_3d



    # Remove any patterns that have duplicates at T=1 

    while True:

        is_deduped = True

        dedup = {}

        for t0_key, pattern_3d in index.items():

            t1_key = np.product([ hash_geometric(board) for board in pattern_3d[1:] ])

            if t1_key not in index:

                dedup[t0_key] = pattern_3d 

            else:

                is_deduped = False

        index = dedup

        if is_deduped: break

        

    # Group by last frame

    grouped  = defaultdict(list)

    for pattern_3d in index.values():

        order_key = hash_geometric(pattern_3d[-1])

        grouped[order_key].append(pattern_3d)

    grouped = { **grouped }  # remove defaultdict

    return grouped

patterns       = generated_patterns(shape=(5,3)) + generated_patterns(shape=(4,4))

patterns_tails = tail_repeating_patterns(patterns)
for solution_3d in patterns_tails: plot_3d(solution_3d)
pattern_groups = grouped_patterns(patterns)

print('len(patterns)',                len(patterns))

print('len(pattern_groups.values())', len(list(chain(*pattern_groups.values()))))

print('len(pattern_groups)',          len(pattern_groups))

for solution_3d in chain(*pattern_groups.values()): plot_3d(solution_3d)