from collections import defaultdict

from fastcache import clru_cache

from joblib import Parallel

from joblib import delayed

# from mergedeep import merge

from numba import njit, prange

from scipy.signal import convolve2d

from typing import Union, List, Tuple, Dict, Callable

from itertools import chain, product



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

import skimage

import skimage.measure

import pydash



notebook_start = time.perf_counter()





### Don't wrap console output text

from IPython.display import display, HTML

display(HTML("""

<style>

div.output_area pre {

    white-space: pre;

    width: 100%;

}

</style>

"""))





%load_ext autoreload

%autoreload 2
# TODO: add z3-solver to kaggle-docker image

! python3 -m pip install -q z3-solver

! apt-get install -qq tree moreutils
# Download git repository and copy to local directory

!rm -rf /ai-games/

!git clone https://github.com/JamesMcGuigan/ai-games/ /ai-games/

!cp -rf /ai-games/puzzles/game_of_life/* ./   # copy code to kaggle notebook

!rm -rf /kaggle/working/neural_networks/      # not relevant to this notebook

!cd /ai-games/; git log -n1 
from utils.util import *

from utils.plot import *

from utils.game import *

from utils.datasets import *

from utils.tuplize import *

from hashmaps.crop import *

from hashmaps.hash_functions import *

from hashmaps.translation_solver import *

from hashmaps.repeating_patterns import *

from constraint_satisfaction.fix_submission import *
import itertools

from itertools import product



import numpy as np





def tessellate_board(board):

    """ Create a 75x75 (3x) tesselation of the board to account for edge objects """

    shape        = board.shape

    tessellation = np.zeros((shape[0]*3, shape[1]*3), dtype=np.int8)

    for x,y in product( range(3), range(3) ):

        tessellation[ shape[0]*x : shape[0]*(x+1), shape[1]*y : shape[1]*(y+1) ] = board

    return tessellation





def detessellate_board(tessellation):

    """ Merge 3x tesselation back into 25x25 grid, by working out sets of overlapping regions """

    shape = tessellation.shape[0] // 3, tessellation.shape[1] // 3

    views = np.stack([

        tessellation[ shape[0]*x : shape[0]*(x+1), shape[1]*y : shape[1]*(y+1) ].flatten()

        for x,y in product( range(3), range(3) )

    ])

    cells = [ set(views[:,n]) - {0} for n in range(len(views[0])) ]

    for cell1, cell2 in itertools.product(cells, cells):

        if cell1 & cell2:

            cell1 |= cell2  # merge overlapping regions

            cell2 |= cell1

    cells  = np.array([ min(cell) if cell else 0 for cell in cells ])

    labels = sorted(set(cells))

    cells  = np.array([ labels.index(cell) for cell in cells ])  # map back to sequential numbers

    return cells.reshape(shape)

from typing import List



import numpy as np

import scipy

import scipy.ndimage

import scipy.sparse

import skimage

import skimage.measure



# from image_segmentation.tessellation import detessellate_board

# from image_segmentation.tessellation import tessellate_board





def label_board(board):

    """  """

    tessellation = tessellate_board(board)

    tessellation = scipy.ndimage.convolve(tessellation, [[0,1,0],[1,1,1],[0,1,0]]).astype(np.bool).astype(np.int8)

    labeled = skimage.measure.label(tessellation, background=0, connectivity=2)

    labeled = detessellate_board(labeled)

    return labeled





def extract_clusters(board: np.ndarray) -> List[np.ndarray]:

    labeled  = label_board(board)

    return extract_clusters_from_labels(board, labeled)





def extract_clusters_from_labels(board: np.ndarray, labeled: np.ndarray) -> List[np.ndarray]:

    labels   = np.unique(labeled)

    clusters = []

    for label in labels:

        # if label == 0: continue  # preserve index order with labels

        cluster = board * ( labeled == label )

        clusters.append(cluster)

    return clusters
for n, board in enumerate([

    csv_to_numpy(test_df,  50022, key='stop'),

    csv_to_numpy(train_df, 43612, key='stop'),

    csv_to_numpy(train_df, 22282, key='stop'),

    csv_to_numpy(test_df,  90081, key='stop'),

]):

    clusters = extract_clusters(board)



    plt.figure(figsize=((len(clusters)+2)*4, 4))

    plt.subplot(1, 2+len(clusters), 1)

    plt.imshow(board, cmap='binary')

    plt.subplot(1, 2+len(clusters), 2)

    plt.imshow(label_board(board))

    

    for n, cluster in enumerate(clusters):

        plt.subplot(1, 2+len(clusters), n+3)

        plt.imshow(cluster, cmap='binary')
from collections import defaultdict



import pydash

from joblib import delayed

from joblib import Parallel



from hashmaps.crop import filter_crop_and_center

from hashmaps.hash_functions import hash_geometric

# from image_segmentation.clusters import extract_clusters

from utils.game import life_step_3d





def get_cluster_history_lookup(boards, forward_play=10):

    """

    return history[now_hash][delta][past_hash] = {

        "start": past_cluster,

        "stop":  now_cluster,

        "delta": delta,

        "count": 1

    }

    """

    history  = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    clusters = Parallel(-1)( delayed(extract_clusters)(board)       for board in boards )

    clusters = Parallel(-1)( delayed(filter_crop_and_center)(board) for board in pydash.flatten(clusters) )

    clusters = [ cluster for cluster in clusters if cluster is not None ]

    hashes   = Parallel(-1)( delayed(hash_geometric)(board)         for board in clusters )

    clusters = { hashed: cluster for hashed, cluster in zip(hashes, clusters) }  # dedup

    for cluster in clusters.values():

        futures = life_step_3d(cluster, forward_play)

        hashes  = Parallel(-1)( delayed(hash_geometric)(future) for future in futures )

        for t in range(1, forward_play+1):

            past_cluster = futures[t]

            past_hash    = hashes[t]

            for delta in range(1,5+1):

                if t + delta >= len(futures): continue

                now_cluster = futures[t + delta]

                now_hash    = hashes[t + delta]

                if not past_hash in history[now_hash][delta]:

                    history[now_hash][delta][past_hash] = {

                        "start": past_cluster,

                        "stop":  now_cluster,

                        "delta": delta,

                        "count": 1

                    }

                else:

                    history[now_hash][delta][past_hash]['count'] += 1





    # remove defaultdict and sort by count

    history = { now_hash: { delta: dict(sorted(d2.items(), key=lambda pair: pair[1]['count'], reverse=True ))

                for delta,     d2 in d1.items()      }

                for now_hash,  d1 in history.items() }



    # Remove any past boards with less than half the frequency of the most common board

    for now_hash, d1 in history.items():

        for delta, d2 in d1.items():

            max_count = max([ values['count'] for values in d2.values() ])

            for past_hash, values in list(d2.items()):

                if values['count'] < max_count/2: del history[now_hash][delta][past_hash]

    return history

# Only process a small dataset size for debugging when running in Interactive mode, else process everything  

dataset_size = 100 if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive' else 400_000

print(f'dataset_size = {dataset_size}')
import gzip

import os

import pickle

from typing import Any



import humanize





def read_gzip_pickle_file(filename: str) -> Any:

    try:

        if not os.path.exists(filename): raise FileNotFoundError

        with open(filename, 'rb') as file:

            data = file.read()

            try:    data = gzip.decompress(data)

            except: pass

            data = pickle.loads(data)

    except Exception as exception:

        data = None

    return data





def save_gzip_pickle_file(data: Any, filename: str, verbose=True) -> int:

    try:

        with open(filename, 'wb') as file:

            data = pickle.dumps(data)

            data = gzip.compress(data)

            file.write(data)

            file.close()

        filesize = os.path.getsize(filename)

        if verbose: print(f'wrote: {filename} = {humanize.naturalsize(filesize)}')

        return filesize

    except:

        return 0
# Source: https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver

import gzip

import os

import pickle

import time



import humanize

import numpy as np



# from image_segmentation.history_lookup import get_cluster_history_lookup

# from utils.datasets import output_directory

# from utils.datasets import test_df

# from utils.datasets import train_df

# from utils.game import generate_random_boards

# from utils.gzip_pickle_file import read_gzip_pickle_file

# from utils.gzip_pickle_file import save_gzip_pickle_file

# from utils.util import csv_to_numpy_list





def generate_cluster_history_lookup(dataset_size=250_000, verbose=True):

    time_start = time.perf_counter()



    csv_size = len(train_df.index) # + len(test_df.index)

    dataset = np.concatenate([

        csv_to_numpy_list(train_df, key='start'),

        # csv_to_numpy_list(test_df,  key='stop'),

        generate_random_boards(max(1, dataset_size - csv_size))

    ])[:dataset_size]

    cluster_history_lookup = get_cluster_history_lookup(dataset, forward_play=10)



    time_taken = time.perf_counter() - time_start

    if verbose: print(f'{len(cluster_history_lookup)} unique clusters in {time_taken:.1f}s = {1000*time_taken/len(dataset):.0f}ms/board')

    return cluster_history_lookup







cluster_history_lookup_cachefile = f'{output_directory}/cluster_history_lookup.pickle'

cluster_history_lookup = read_gzip_pickle_file(cluster_history_lookup_cachefile)



if __name__ == '__main__':

    cluster_history_lookup = generate_cluster_history_lookup(dataset_size=dataset_size)

    save_gzip_pickle_file(cluster_history_lookup, cluster_history_lookup_cachefile)

import time



import numpy as np

from joblib import delayed

from joblib import Parallel



from constraint_satisfaction.fix_submission import is_valid_solution

from hashmaps.hash_functions import hash_geometric

from hashmaps.translation_solver import solve_translation

# from image_segmentation.clusters import extract_clusters_from_labels

# from image_segmentation.clusters import label_board

# from image_segmentation.history_lookup_cache import cluster_history_lookup

from utils.datasets import sample_submission_df

from utils.util import csv_to_delta_list

from utils.util import csv_to_numpy_list

from utils.util import numpy_to_series





def image_segmentation_dataframe_solver( df, history, submission_df=None, exact=False, blank_missing=True, verbose=True ):

    time_start = time.perf_counter()

    stats      = { "partial": 0, "exact": 0, "total": 0 }



    submission_df = submission_df if submission_df is not None else sample_submission_df.copy()

    idxs       = df.index

    deltas     = csv_to_delta_list(df)

    boards     = csv_to_numpy_list(df, key='stop')

    labeleds   = Parallel(-1)( delayed(label_board)(board)                          for board in boards )

    clustereds = Parallel(-1)( delayed(extract_clusters_from_labels)(board, labels) for board, labels in zip(boards, labeleds) )



    for idx, delta, stop_board, labels, clusters in zip(idxs, deltas, boards, labeleds, clustereds):

        start_board = image_segmentation_solver(

            stop_board, delta, history=history, blank_missing=blank_missing,

            labels=labels, clusters=clusters

        )



        is_valid = is_valid_solution( start_board, stop_board, delta )

        if   is_valid:                         stats['exact']   += 1

        elif np.count_nonzero( start_board ):  stats['partial'] += 1

        stats['total'] += 1



        if is_valid or not exact:

            submission_df.loc[idx] = numpy_to_series(start_board, key='start')





    time_taken = time.perf_counter() - time_start

    stats['time_seconds'] = int(time_taken)

    stats['time_hours']   = round(time_taken/60/60, 2)

    if verbose: print('image_segmentation_solver()', stats)

    return submission_df







def image_segmentation_solver(stop_board, delta, history=None, blank_missing=True, labels=None, clusters=None):

    history  = history  if history  is not None else cluster_history_lookup

    labels   = labels   if labels   is not None else label_board(stop_board)

    clusters = clusters if clusters is not None else extract_clusters_from_labels(stop_board, labels)



    labels       = np.unique(labels)

    now_hashes   = Parallel(-1)( delayed(hash_geometric)(cluster) for cluster in clusters )

    new_clusters = {}

    for label, now_cluster, now_hash in zip(labels, clusters, now_hashes):

        if label == 0: continue

        if np.count_nonzero(now_cluster) == 0: continue

        if history.get(now_hash,{}).get(delta,None):

            for past_hash in history[now_hash][delta].keys():  # sorted by count

                try:

                    start_cluster = history[now_hash][delta][past_hash]['start']

                    stop_cluster  = history[now_hash][delta][past_hash]['stop']

                    transform_fn  = solve_translation(stop_cluster, now_cluster) # assert np.all( transform_fn(train_board) == test_board )

                    past_cluster  = transform_fn(start_cluster)

                    new_clusters[label] = past_cluster

                    break

                except Exception as exception:

                    pass

        if not label in new_clusters:

            if blank_missing: new_clusters[label] = np.zeros(now_cluster.shape, dtype=np.int8)

            else:             new_clusters[label] = now_cluster



    # TODO: return list of all possible cluster permutations

    start_board = np.zeros( stop_board.shape, dtype=np.int8 )

    for cluster in new_clusters.values():

        start_board += cluster

    start_board = start_board.astype(np.bool).astype(np.int8)

    return start_board
submission_df = image_segmentation_dataframe_solver( test_df[:dataset_size], history=cluster_history_lookup, exact=False )

submission_df.to_csv('submission.csv')
# Count number of non-zero entries in each submission.csv file

!( for FILE in $(find ./ ../input/ -name 'submission.csv' | sort ); do cat $FILE | grep ',1' | wc -l | tr '\n' ' '; echo $FILE; done) | sort -n;



# Merge submission files from various sources into a single file. Reverse sort puts non-zero entries first, then use awk to deduplicate on id

!find ./ ../input/ -name 'submission.csv' | xargs cat | sort -nr | uniq | awk -F',' '!a[$1]++' | sort -n > ./submission.csv



# Count number of non-zero entries in each submission.csv file

!( for FILE in $(find ./ ../input/ -name 'submission.csv' | sort ); do cat $FILE | grep ',1' | wc -l | tr '\n' ' '; echo $FILE; done) | sort -n;



# BUGFIX: previous version of the code was computing to delta=-1, so replay submission.csv forward one step if required and validate we have the correct delta

# This also generates stats

!PYTHONPATH='.' python3 ./constraint_satisfaction/fix_submission.py