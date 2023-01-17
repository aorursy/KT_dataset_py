import sys

import math

import time

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from numba import njit

from fastcache import clru_cache
def generate_primes(count):

    primes = [2]

    for n in range(3, sys.maxsize, 2):

        if len(primes) >= count: break

        if all( n % i != 0 for i in range(3, int(math.sqrt(n))+1, 2) ):

            primes.append(n)

    return primes



primes_np = np.array( generate_primes(625), dtype=np.int64 )  # first 625 primes 





# 50 Summable Primes

# with the lesser constraint that: 

#     no 2-pair of prime numbers in the set has the same sum (rather than any combination)

# This works in practice, but is not gaurenteed prevent hash collisions

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
# Hashable Primes: 2, 7, 23, 47, 61

geometric_hash_pattern = np.array([

    [0,0,0,0,61,0,0,0,0],

    [0,0,0,0,47,0,0,0,0],

    [0,0,0,0,23,0,0,0,0],

    [0,0,0,0, 7,0,0,0,0],

 [61,47,23,7, 2,7,23,47,61],

    [0,0,0,0, 7,0,0,0,0],

    [0,0,0,0,23,0,0,0,0],

    [0,0,0,0,47,0,0,0,0],

    [0,0,0,0,61,0,0,0,0],

])

transforms = {

    'identity': geometric_hash_pattern,

    'flip':     np.flip(geometric_hash_pattern),

    'rot90':    np.rot90(geometric_hash_pattern),

    'roll':     np.roll(np.roll(geometric_hash_pattern, -2, axis=0), -1, axis=1),    

}



plt.figure(figsize=(len(transforms)*5, 5))

for i, (name, grid) in enumerate(transforms.items()):

    plt.subplot(1, len(transforms), i+1)

    plt.title(name)

    plt.imshow(grid, cmap='nipy_spectral')
train_file = f'../input/conways-reverse-game-of-life-2020/train.csv'

test_file  = f'../input/conways-reverse-game-of-life-2020/test.csv'



train_df   = pd.read_csv(train_file, index_col='id')

test_df    = pd.read_csv(test_file,  index_col='id')



def csv_to_numpy_list(df, key='start') -> np.ndarray:

    return df[[ f'{key}_{n}' for n in range(25**2) ]].values.reshape(-1,25,25)
def test_hash_geometric():

    boards = csv_to_numpy_list(train_df, key='start')

    for board in boards:

        transforms = {

            "identity": board,

            "roll_0":   np.roll(board, 1, axis=0),

            "roll_1":   np.roll(board, 1, axis=1),

            "flip_0":   np.flip(board, axis=0),

            "flip_1":   np.flip(board, axis=1),

            "rot90":    np.rot90(board, 1),

            "rot180":   np.rot90(board, 2),

            "rot270":   np.rot90(board, 3),

        }

        hashes = { f'{key:8s}': hash_geometric(value) for key, value in transforms.items()}



        # all geometric transforms should produce the same hash

        assert len(set(hashes.values())) == 1

    return len(boards)

        

count = test_hash_geometric()

print(f'{count} tests passed')
boards = csv_to_numpy_list(train_df, key='start')



time_start = time.perf_counter()

hashes = [ hash_geometric(board) for board in boards ]

time_taken = time.perf_counter() - time_start



print(f'hash_geometric() = {1000 * time_taken/len(boards):0.3f}ms')
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
# Hashable Primes: 2, 7, 23, 47, 61, 83, 131, 163, 173, 251

translation_hash_pattern = np.array([

       [0,0,0,0, 83,0,0,0,0],

       [0,0,0,0,131,0,0,0,0],

       [0,0,0,0,163,0,0,0,0],

       [0,0,0,0,173,0,0,0,0],

[83,131,163,173,  2,7,23,47,61],

       [0,0,0,0,  7,0,0,0,0],

       [0,0,0,0, 23,0,0,0,0],

       [0,0,0,0, 47,0,0,0,0],

       [0,0,0,0, 61,0,0,0,0],

])

transforms = {

    'identity': translation_hash_pattern,

    'flip':     np.flip(translation_hash_pattern),

    'rot90':    np.rot90(translation_hash_pattern),

    'roll':     np.roll(np.roll(translation_hash_pattern, -2, axis=0), -1, axis=1),    

}



plt.figure(figsize=(len(transforms)*5, 5))

for i, (name, grid) in enumerate(transforms.items()):

    plt.subplot(1, len(transforms), i+1)

    plt.title(name)

    plt.imshow(grid, cmap='nipy_spectral')
board = csv_to_numpy_list(test_df.loc[85291], key='stop')[0]

for transforms in [

    {

        "identity": board,

        "np.roll(0)":   np.roll(board, 10, axis=0),

        "np.roll(1)":   np.roll(board, 10, axis=1),

        "np.rot180()":  np.rot90(board, 2),

    },

    {

        "np.flip(0)":   np.flip(board, axis=0),

        "np.flip(1)":   np.flip(board, axis=1),

        "np.rot90()":   np.rot90(board, 1),

        "np.rot180()":  np.rot90(board, 2),

        "np.rot270()":  np.rot90(board, 3),

    }    

]:

    figure = plt.figure(figsize=(len(transforms)*5, 5*2))

    figure.tight_layout(pad=10.0)

    for i, (name, grid) in enumerate(transforms.items()):

        plt.subplot(2, len(transforms), i+1)

        plt.title(name)

        plt.imshow(grid, cmap = 'binary')



    for i, (name, grid) in enumerate(transforms.items()):

        hashmap          = hash_translations_board(grid)

        sum_hash         = np.sum(hashmap)

        sum_x_prime_hash = hash_translations(grid)



        plt.subplot(2, len(transforms), len(transforms) + i+1)

        plt.title(f'sum = {sum_hash}\nsum * primes = {sum_x_prime_hash}')

        plt.imshow(hashmap, cmap = 'nipy_spectral')
def test_hash_translations():

    boards = csv_to_numpy_list(train_df, key='start')

    for board in boards:

        if np.count_nonzero(board) < 50: continue  # skip small symmetric boards

        transforms = {

            "identity": board,

            "roll_0":   np.roll(board, 13, axis=0),

            "roll_1":   np.roll(board, 13, axis=1),

            "flip_0":   np.flip(board, axis=0),

            "flip_1":   np.flip(board, axis=1),

            "rot90":    np.rot90(board, 1),

            "rot180":   np.rot90(board, 2),

            "rot270":   np.rot90(board, 3),

        }

        hashes  = { key: hash_translations(value) for key, value in transforms.items()  }



        # rolling the board should not change the hash, but other transforms should

        assert hashes['identity'] == hashes['roll_0']

        assert hashes['identity'] == hashes['roll_1']



        # all other flip / rotate transformations should produce different hashes

        assert hashes['identity'] != hashes['flip_0']

        assert hashes['identity'] != hashes['flip_1']

        assert hashes['identity'] != hashes['rot90']

        assert hashes['identity'] != hashes['rot180']

        assert hashes['identity'] != hashes['rot270']

        assert hashes['flip_0'] != hashes['flip_1'] != hashes['rot90']  != hashes['rot180'] != hashes['rot270']

    return len(boards)



count = test_hash_translations()

print(f'{count} tests passed')
boards = csv_to_numpy_list(train_df, key='start')



time_start = time.perf_counter()

hashes = [ hash_translations(board) for board in boards ]

time_taken = time.perf_counter() - time_start



print(f'hash_translations() = {1000 * time_taken/len(boards):0.3f}ms')