# TODO: add z3-solver to kaggle-docker image

! python3 -m pip install -q z3-solver

! apt-get install -qq tree moreutils
# Download git repository and copy to local directory

!rm -rf /ai-games/

!git clone https://github.com/JamesMcGuigan/ai-games/ /ai-games/

# !cd /ai-games/; git checkout ad2f8cc94865f1be6083ca699d4b62b0cc039435

!cp -rf /ai-games/puzzles/game_of_life/* ./   # copy code to kaggle notebook

!rm -rf /kaggle/working/neural_networks/      # not relevant to this notebook

!cd /ai-games/; git log -n1 
! pwd

! tree -F
# Merge submission files from various sources into a single file. Reverse sort puts non-zero entries first, then use awk to deduplicate on id

!find ./ ../input/ /ai-games/puzzles/game_of_life/ -name 'submission.csv' | xargs cat | sort -nr | uniq | awk -F',' '!a[$1]++' | sort -n > ./submission_previous.csv

!find ./ ../input/ /ai-games/puzzles/game_of_life/ -name 'submission.csv' | xargs cat | sort -nr | uniq | awk -F',' '!a[$1]++' | sort -n > ./submission.csv

!find ./ ../input/ /ai-games/puzzles/game_of_life/ -name 'timeouts.csv'   | xargs cat | sort -nr | uniq | awk -F',' '!a[$1]++' | sort -n > ./timeouts.csv



# Count number of non-zero entries in each submission.csv file

!( for FILE in $(find ./ ../input/ /ai-games/puzzles/game_of_life/ -name '*submission.csv' | sort ); do cat $FILE | grep ',1' | wc -l | tr '\n' ' '; echo $FILE; done) | sort -n;



# BUGFIX: previous version of the code was computing to delta=-1, so replay submission.csv forward one step if required and validate we have the correct delta

# This also generates stats

!PYTHONPATH='.' python3 ./constraint_satisfaction/fix_submission.py
%load_ext autoreload

%autoreload 2



import itertools

import time

import numpy as np

import pandas as pd

import pydash

from typing import Union, List, Tuple

import os

import sys

from pathos.multiprocessing import ProcessPool



from utils.plot import *

from utils.game import *

from utils.util import *

from utils.datasets import *

from utils.display_source import *



print('os.cpu_count()', os.cpu_count())

notebook_start = time.perf_counter()
# These imports won't work inside Kaggle Submit without an internet connection to install Z3

import z3

from constraint_satisfaction.z3_solver import *

from constraint_satisfaction.solve_dataframe import *
display_source('./constraint_satisfaction/z3_solver.py')
display_source('./constraint_satisfaction/z3_constraints.py')
display_source('./constraint_satisfaction/solve_dataframe.py')
display_source('./utils/idx_lookup.py')
display_source('./utils/game.py')
display_source('./utils/util.py')
display_source('./utils/plot.py')
display_source('./utils/datasets.py')
display_source('./test/test_submission.py')
test_df
idx      = 0  

delta    = csv_to_delta(train_df, idx)

board    = csv_to_numpy(train_df, idx, key='stop')

expected = csv_to_numpy(train_df, idx, key='start')



time_start     = time.perf_counter()

solution_count = 0

z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta, idx=idx)

while np.count_nonzero(solution_3d):

    solution_count += 1

    plot_3d(solution_3d)

    z3_solver, t_cells, solution_3d = game_of_life_next_solution(z3_solver, t_cells, verbose=True) # takes ~0.5s per solution

    if solution_count > 5: break

# print(f'Total Solutions: {solution_count} in {time.perf_counter() - time_start:.1f}s')  # too many to count
idx      = 0

delta    = csv_to_delta(train_df, idx)

board    = csv_to_numpy(train_df, idx, key='stop')

solution_3d, idx, time_taken = solve_board_delta1_loop(board, delta, idx)

plot_3d( solution_3d )
# test_df.loc[72539] | delta = 1 | time = 10.4s

# test_df.loc[56795] | delta = 1 | time = 9.5s

# test_df.loc[58109] | delta = 2 | time = 21.1s

# test_df.loc[62386] | delta = 2 | time = 20.5s

# test_df.loc[64934] | delta = 3 | time = 41.4s

# test_df.loc[77908] | delta = 3 | time = 49.5s

# test_df.loc[55567] | delta = 4 | time = 151.8s

# test_df.loc[71076] | delta = 4 | time = 239.0s

# test_df.loc[74622] | delta = 5 | time = 2119.0s

# test_df.loc[75766] | delta = 5 | time = 1518.6s



import random

from joblib import delayed

from joblib import Parallel

submision_df = pd.read_csv('submission.csv', index_col='id')  # manually copy/paste sample_submission.csv to location



job_idxs = []

delta_batch_size = os.cpu_count()

delta_idxs = {}

for delta in [1,2,3]:  # sorted(test_df['delta'].unique()):  # [1,2,3,4,5]

    df   = test_df

    df   = df[ df['delta'] == delta ]                                 # smaller deltas are exponentially easier

    df   = df.iloc[ df.apply(np.count_nonzero, axis=1).argsort()  ]   # smaller grids are easiest 

    idxs = get_unsolved_idxs(df, submision_df, modulo=(10,1) )        # don't recompute solved boards

    idxs = idxs[:delta_batch_size]                                    # sample of small boards 

    delta_idxs[delta] = idxs

    # quartiles from each delta - this takes far too long

    # idxs = [ idxs[0], idxs[len(idxs)*1//4], idxs[len(idxs)*1//2], idxs[len(idxs)*3//4], idxs[-1] ] 

    job_idxs += idxs

print('delta_idxs', delta_idxs)

    

jobs = []    

df   = test_df

for n, idx in enumerate(job_idxs):

    delta = csv_to_delta(df, idx)

    board = csv_to_numpy(df, idx, key='stop')

    def job_fn(board, delta, idx):

        time_start = time.perf_counter()

        z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta, idx=idx, verbose=True)

        time_taken = time.perf_counter() - time_start

        return solution_3d, idx, delta, time_taken 

    jobs.append( delayed(job_fn)(board, delta, idx) )

        

jobs = []       

# pathos.multiprocessing: Pool.uimap() is used for the submission.csv loop as it uses iterator rather than batch semantics  

jobs_output = Parallel(-1)(reversed(jobs))      # run longest jobs first 

for solution_3d, idx, delta, time_taken in reversed(jobs_output):

    print(f'test_df.loc[{idx}] | delta = {delta} | cells = {np.count_nonzero(solution_3d[-1])} | time = {time_taken:.1f}s')

    if is_valid_solution_3d(solution_3d):        

        plot_3d(solution_3d)

        # Save to submission.csv.csv file         

        submision_df          = pd.read_csv('submission.csv', index_col='id')

        solution_dict         = numpy_to_dict(solution_3d[0])

        submision_df.loc[idx] = pd.Series(solution_dict)

        submision_df.sort_index().to_csv('submission.csv')

    else:

        print('Unsolveable!')

        plot_idx(test_df, idx)
notebook_time = (time.perf_counter() - notebook_start)

print(f'notebook_time = {notebook_time:.0f}s = {notebook_time/60:.1f}m')
submission_df = solve_dataframe(test_df, savefile='submission.csv', timeout=(8*60*60 - notebook_time), modulo=(9,1), plot=True)

# submission_df = solve_dataframe(test_df, savefile='submission.csv', timeout=(1*60*60 - notebook_time), modulo=(9,1), plot=True)
# Cleanup python caches to prevent poluting kaggle notebook outputs

!find ./ -name '__pycache__' -or -name '*.py[cod]'  -delete
# BUGFIX: previous version of the code was computing to delta=-1, so replay submission.csv forward one step if required and validate we have the correct delta

# This also generates stats

!PYTHONPATH='.' python3 ./constraint_satisfaction/fix_submission.py
# Count number of non-zero entries in each submission.csv file

!( for FILE in *.csv; do cat $FILE | grep ',1,' | wc -l | tr '\n' ' '; echo $FILE; done ) | sort -n;