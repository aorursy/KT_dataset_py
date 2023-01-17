# !pip3 install -q mergedeep
from collections import defaultdict

from fastcache import clru_cache

from joblib import Parallel

from joblib import delayed

# from mergedeep import merge

from numba import njit, prange

from scipy.signal import convolve2d

from typing import Union, List, Tuple, Dict, Callable



import humanize

import itertools

import math

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import scipy

import scipy.sparse

import sys

import time
notebook_start = time.perf_counter()
!ls -la ../input
train_df             = pd.read_csv('../input/conways-reverse-game-of-life-2020/train.csv', index_col='id')

test_df              = pd.read_csv('../input/conways-reverse-game-of-life-2020/test.csv',  index_col='id')

sample_submission_df = pd.read_csv('../input/conways-reverse-game-of-life-2020/sample_submission.csv',  index_col='id')

submission_df        = pd.read_csv('../input/game-of-life-hashmap-solver/submission.csv',  index_col='id')
@clru_cache(None)

def csv_column_names(key='start'):

    return [ f'{key}_{n}' for n in range(25**2) ]





def csv_to_delta(df, idx):

    return int(df.loc[idx]['delta'])



def csv_to_delta_list(df):

    return df['delta'].values





def csv_to_numpy(df, idx, key='start') -> np.ndarray:

    try:

        columns = csv_column_names(key)

        board   = df.loc[idx][columns].values

    except:

        board = np.zeros((25, 25))

    board = board.reshape((25,25)).astype(np.int8)

    return board





def csv_to_numpy_list(df, key='start') -> np.ndarray:

    try:

        columns = csv_column_names(key)

        output  = df[columns].values.reshape(-1,25,25)

    except:

        output  = np.zeros((0,25,25))

    return output





# noinspection PyTypeChecker,PyUnresolvedReferences

def numpy_to_dict(board: np.ndarray, key='start') -> Dict:

    assert len(board.shape) == 2  # we want 2D solutions_3d[0] not 3D solutions_3d

    assert key in { 'start', 'stop' }



    board  = np.array(board).flatten().tolist()

    output = { f"{key}_{n}": board[n] for n in range(len(board))}

    return output





def numpy_to_series(board: np.ndarray, key='start') -> pd.Series:

    solution_dict = numpy_to_dict(board, key)

    return pd.Series(solution_dict)
# Functions for implementing Game of Life Forward Play



# Source: https://www.kaggle.com/ianmoone0617/reversing-conways-game-of-life-tutorial

def life_step_1(X: np.ndarray):

    """Game of life step using generator expressions"""

    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)

                     for i in (-1, 0, 1) for j in (-1, 0, 1)

                     if (i != 0 or j != 0))

    return (nbrs_count == 3) | (X & (nbrs_count == 2))





# Source: https://www.kaggle.com/ianmoone0617/reversing-conways-game-of-life-tutorial

def life_step_2(X: np.ndarray):

    """Game of life step using scipy tools"""

    from scipy.signal import convolve2d

    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X

    return (nbrs_count == 3) | (X & (nbrs_count == 2))







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

def life_neighbours(board: np.ndarray, max_value=3):

    size_x = board.shape[0]

    size_y = board.shape[1]

    output = np.zeros(board.shape, dtype=np.int8)

    for x in range(size_x):

        for y in range(size_y):

            output[x,y] = life_neighbours_xy(board, x, y, max_value)

    return output





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
def plot_3d(solution_3d: np.ndarray, size=4, max_cols=6):

    cols = np.min([ len(solution_3d), max_cols ])

    rows = len(solution_3d) // cols + 1

    plt.figure(figsize=(cols*size, rows*size))

    for t in range(len(solution_3d)):

        board = solution_3d[t]

        plt.subplot(rows, cols, t + 1)

        plt.imshow(board, cmap='binary'); plt.title(f't={t}')

    plt.show()
# Reference: The First 10,000 Primes - https://primes.utm.edu/lists/small/10000.txt

def generate_primes(count):

    primes = [2]

    for n in range(3, sys.maxsize, 2):

        if len(primes) >= count: break

        if all( n % i != 0 for i in range(3, int(math.sqrt(n))+1, 2) ):

            primes.append(n)

    return primes



primes     = generate_primes(10_000)

primes_np  = np.array(primes, dtype=np.int64)

primes_set = set(primes_np)



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
def test_hash_geometric():

    for idx in range(1000):

        board = csv_to_numpy(train_df, idx)

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





def test_hash_translations():

    for idx in range(1000):

        board = csv_to_numpy(train_df, idx)

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

        

        

test_hash_geometric()

test_hash_translations()

print('All Tests Passed!')
from joblib import delayed

from joblib import Parallel



def hashmap_dataframe(df: pd.DataFrame, key='start'):

    boards                = csv_to_numpy_list(df, key=key)

    geometric_hashes      = Parallel(-1)( delayed(hash_geometric)(board)    for board in boards )

    translation_hashes    = Parallel(-1)( delayed(hash_translations)(board) for board in boards )



    output = df.copy(deep=True)

    output['id']                      = df.index

    output[f'{key}_geometric_hash']   = geometric_hashes

    output[f'{key}_translation_hash'] = translation_hashes



    output = output.astype('int64')

    output = output.astype({ col: 'int8' for col in csv_column_names(key) })

    return output
%%time

hashmap_train_df = train_df

hashmap_test_df  = test_df 

hashmap_train_df = hashmap_dataframe(hashmap_train_df, key='start')

hashmap_train_df = hashmap_dataframe(hashmap_train_df, key='stop')

hashmap_test_df  = hashmap_dataframe(hashmap_test_df,  key='stop')



hashmap_train_df.to_csv('./hashmap_train.csv')

hashmap_test_df.to_csv('./hashmap_test.csv')
hashmap_train_df
hashmap_test_df
def count_geometric_duplicates():

    # Create hashtable index for train_df

    train_stop_geometric_rows    = defaultdict(list)

    train_stop_translation_rows  = defaultdict(list)

    for idx, train_row in hashmap_train_df.iterrows():

        delta                 = train_row['delta']

        stop_geometric_hash   = train_row['stop_geometric_hash']

        stop_translation_hash = train_row['stop_translation_hash']

        train_stop_geometric_rows[stop_geometric_hash]     += [ train_row ]

        train_stop_translation_rows[stop_translation_hash] += [ train_row ]





    # Now count the number of hash matches in test_df

    count_exact       = 0

    count_geometric   = 0

    count_translation = 0

    count_total       = 0

    for idx, test_row in hashmap_test_df.iterrows():

        delta                      = test_row['delta']

        test_stop_geometric_hash   = test_row['stop_geometric_hash']

        test_stop_translation_hash = test_row['stop_translation_hash']



        count_total += 1



        # See if we find any geometric or translation hash matches

        if test_stop_translation_hash in train_stop_translation_rows:

            count_translation += 1



        if test_stop_geometric_hash in train_stop_geometric_rows:

            count_geometric += 1

            for train_row in train_stop_geometric_rows[test_stop_geometric_hash]:

                if train_row['delta'] == delta:

                    count_exact += 1

                    break



    print(" | ".join([

        f'count_exact = {count_exact} ({100*count_exact/count_total:.1f}%)',

        f'count_geometric = {count_geometric} ({100*count_geometric/count_total:.1f}%)',

        f'count_translation = {count_translation} ({100*count_translation/count_total:.1f}%)',

        f'count_total = {count_total}'

    ]))
def identity(board): return board

def rot90(board):    return np.rot90(board, 1)

def rot180(board):   return np.rot90(board, 2)

def rot270(board):   return np.rot90(board, 3)

def flip(board):     return np.flip(board)

def flip90(board):   return np.flip(np.rot90(board, 1))

def flip180(board):  return np.flip(np.rot90(board, 2))

def flip270(board):  return np.flip(np.rot90(board, 3))

geometric_transforms = [identity, rot90, rot180, rot270, flip, flip90, flip180, flip270]







def solve_geometric(train_board, test_board) -> Callable:

    """

    Find the function required to correctly orientate train_board to match test_board

    This is a simple brute force search over geometric_transforms until matching hash_translations() are found

    """

    assert hash_geometric(train_board) == hash_geometric(test_board)



    geometric_fn = None

    test_hash    = hash_translations(test_board)

    for transform_fn in geometric_transforms:

        train_transform = transform_fn(train_board)

        train_hash      = hash_translations(train_transform)

        if train_hash == test_hash:

            geometric_fn = transform_fn

            break  # we are lazily assuming there will be only one matching function



    assert geometric_fn is not None

    return geometric_fn





def solve_translation(train_board, test_board) -> Callable:

    """

    Find the function required to correctly transform train_board to match test_board

    We compute the sums of cell counts along each axis, then roll them until they match

    """

    train_x_counts = np.count_nonzero(train_board, axis=1)  # == np.array([ np.count_nonzero(train_board[x,:]) for x in range(train_board.shape[0]) ])

    train_y_counts = np.count_nonzero(train_board, axis=0)  # == np.array([ np.count_nonzero(train_board[:,y]) for y in range(train_board.shape[1]) ])

    test_x_counts  = np.count_nonzero(test_board,  axis=1)  # == np.array([ np.count_nonzero(test_board[x,:])  for x in range(test_board.shape[0])  ])

    test_y_counts  = np.count_nonzero(test_board,  axis=0)  # == np.array([ np.count_nonzero(test_board[:,y])  for y in range(test_board.shape[1])  ])

    assert sorted(train_x_counts) == sorted(test_x_counts)

    assert sorted(train_y_counts) == sorted(test_y_counts)



    # This is a little bit inefficient, compared to comparing indexes of max values, but we are not CPU bound

    x_roll_count = None

    for n in range(len(train_x_counts)):

        if np.roll(train_x_counts, n).tobytes() == test_x_counts.tobytes():

            x_roll_count = n

            break



    y_roll_count = None

    for n in range(len(train_y_counts)):

        if np.roll(train_y_counts, n).tobytes() == test_y_counts.tobytes():

            y_roll_count = n

            break



    assert x_roll_count is not None

    assert y_roll_count is not None



    def transform_fn(board):

        return np.roll(np.roll(board, x_roll_count, axis=0), y_roll_count, axis=1)



    assert np.all( transform_fn(train_board) == test_board )

    return transform_fn





def solve_geometric_translation(train_board, test_board) -> Callable:

    geometric_fn    = solve_geometric(train_board, test_board)

    translation_fn  = solve_translation(geometric_fn(train_board), test_board)



    def transform_fn(board):

        return translation_fn( geometric_fn(board) )

    assert np.all( transform_fn(train_board) == test_board )

    return transform_fn

def build_hashmap_database_from_pandas(

        dfs: Union[pd.DataFrame, List[pd.DataFrame]],

        hash_fn: Callable = hash_geometric,

        future_count = 10,

        keys = ['start', 'stop']

):

    boards = extract_boards_from_dataframe(dfs, keys)

    lookup = build_hashmap_database_from_boards(boards, hash_fn=hash_fn, future_count=future_count)

    return lookup





def extract_boards_from_dataframe(dfs: List[pd.DataFrame], keys = ['start', 'stop'] ):

    boards = []

    if not isinstance(dfs, list): dfs = [ dfs ]

    for df in dfs:

        for key in keys:

            if f'{key}_0' not in df.columns: continue     # skip start on test_df

            for board in csv_to_numpy_list(df, key=key):

                if np.count_nonzero(board) == 0: continue  # skip empty boards

                boards.append(board)

    return boards





def build_hashmap_database_from_boards(

        boards: List[np.ndarray],

        hash_fn: Callable = hash_geometric,

        future_count = 10,

        max_delta    = 5,

):

    assert callable(hash_fn)



    hashmap_database = defaultdict(lambda: defaultdict(dict))  # hashmap_database[stop_hash][delta] = { stop: np, start: np, delta: int }

    future_hashes = Parallel(-1)(

        delayed(build_future_hashes)(board, hash_fn, future_count)

        for board in boards

    )

    for futures, hashes in future_hashes:

        for t in range(len(futures)-max_delta):

            for delta in range(1, max_delta+1):

                start_board = futures[t]

                stop_board  = futures[t + delta]

                stop_hash   = hashes[t + delta]

                hashmap_database[stop_hash][delta] = { 'start': start_board, 'stop': stop_board, 'delta': delta }

    return hashmap_database





def build_future_hashes(board, hash_fn, future_count):

    futures = [ board ]

    for _ in range(future_count): futures += [ life_step(futures[-1]) ]

    hashes  = [ hash_fn(board) for board in futures ]

    return futures, hashes





def solve_hashmap_dataframe(hashmap_database=None, submission_df=None, verbose=True):

    solved = 0

    failed = 0

    total  = len(test_df.index)

    hashmap_database = hashmap_database or build_hashmap_database_from_pandas([ train_df, test_df ], hash_fn=hash_geometric)



    submission_df = submission_df if submission_df is not None else sample_submission_df.copy()

    for test_idx in test_df.index:

        delta       = csv_to_delta(test_df, test_idx)

        test_stop   = csv_to_numpy(test_df, test_idx, key='stop')

        stop_hash   = hash_geometric(test_stop)

        train_start = hashmap_database.get(stop_hash, {}).get(delta, {}).get('start', None)

        train_stop  = hashmap_database.get(stop_hash, {}).get(delta, {}).get('stop', None)

        if train_start is None: continue



        try:

            solution = solve_geometric_translation(train_stop, test_stop)(train_start)



            solution_test = solution

            for t in range(delta): solution_test = life_step(solution_test)

            assert np.all( solution_test == test_stop )



            submission_df.loc[test_idx] = numpy_to_series(solution)

            solved += 1

        except:

            failed += 1



    if verbose:

        print(f'solved = {solved} ({100*solved/total:.1f}%) | failed = {failed} ({100*failed/(solved+failed):.1f}%)')



    return submission_df
%%time

hashmap_database = build_hashmap_database_from_pandas([ train_df, test_df ], hash_geometric)

print('len(hashmap_database) = ', len(hashmap_database))
%%time

submission_df = solve_hashmap_dataframe(hashmap_database, submission_df=submission_df, verbose=True)

submission_df.to_csv('./submission.csv')
# RULES: https://www.kaggle.com/c/conway-s-reverse-game-of-life/data

def generate_random_board():

    # An initial board was chosen by filling the board with a random density between 1% full (mostly zeros) and 99% full (mostly ones).

    # DOCS: https://cmdlinetips.com/2019/02/how-to-create-random-sparse-matrix-of-specific-density/

    density = np.random.random() * 0.98 + 0.01  

    board   = scipy.sparse.random(25, 25, density=density, data_rvs=np.ones).toarray()

    

    # The starting board's state was recorded after the 5 "warmup steps". These are the values in the start variables.

    for t in range(5): board = life_step(board)

    return board



def generate_random_boards(count):

    generated_boards = Parallel(-1)( delayed(generate_random_board)() for _ in range(count) )

    return generated_boards 
%%time 

generated_boards   = generate_random_boards(1_000)

generated_database = build_hashmap_database_from_boards(generated_boards, hash_geometric)

_                  = solve_hashmap_dataframe(generated_database, verbose=True)
%%time 

generated_boards   = generate_random_boards(10_000)

generated_database = build_hashmap_database_from_boards(generated_boards, hash_geometric)

_                  = solve_hashmap_dataframe(generated_database, verbose=True)
%%time 

generated_boards   = generate_random_boards(100_000)

generated_database = build_hashmap_database_from_boards(generated_boards, hash_geometric)

_                  = solve_hashmap_dataframe(generated_database, verbose=True)
# Keep generating new sets of random boards, and hope this cancels out any hash collisions and finds new boards

batch_size = 250_000

endtime    = time.perf_counter() + 8*60*60 - notebook_start

time_start = time.perf_counter()



count = 0

while time.perf_counter() < endtime:

    count             += batch_size

    generated_boards   = generate_random_boards(batch_size)

    generated_database = build_hashmap_database_from_boards(generated_boards, hash_geometric)

    submission_df      = solve_hashmap_dataframe(generated_database, submission_df=submission_df, verbose=False)

    submission_df.to_csv('./submission.csv')

    

time_taken = time.perf_counter() - time_start

print(f'Generated {humanize.intword(count)} boards in {humanize.naturaldelta(time_taken)} = {1000 * time_taken/count:.3f}ms/board')
# Count number of non-zero entries in each submission.csv file

!( for FILE in 'submission.csv'; do cat $FILE | grep ',1,' | wc -l | tr '\n' ' '; echo $FILE; done) | sort -n;