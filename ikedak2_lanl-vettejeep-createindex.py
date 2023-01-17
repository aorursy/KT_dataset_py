import os

import time

import warnings

import traceback

import numpy as np

import pandas as pd

from scipy import stats

import scipy.signal as sg

import multiprocessing as mp

from scipy.signal import hann

from scipy.signal import hilbert

from scipy.signal import convolve

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler



import xgboost as xgb

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



from tqdm import tqdm

warnings.filterwarnings("ignore")
OUTPUT_DIR = ''

DATA_DIR = '../input/'



SIG_LEN = 150000

NUM_SEG_PER_PROC = 4000

NUM_THREADS = 6



NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.

CUTOFF = 18000

MAX_FREQ_IDX = 20000

FREQ_STEP = 2500
# 6x4000のIndexを作成する

def build_rnd_idxs():

    rnd_idxs = np.zeros(shape=(NUM_THREADS, NUM_SEG_PER_PROC), dtype=np.int32)

    max_start_idx = 104982580 - 150000 -10  # V2



    for i in range(NUM_THREADS):

        np.random.seed(5591 + i)

        # 4000個のランダムな整数を作成

        start_indices = np.random.randint(0, max_start_idx, size=NUM_SEG_PER_PROC, dtype=np.int32)

        # 各スレッドのインデックスとして代入

        rnd_idxs[i, :] = start_indices



    # 最初の8個と最後の8個、最大、最小を表示

    for i in range(NUM_THREADS):

        print(rnd_idxs[i, :8])

        print(rnd_idxs[i, -8:])

        print(min(rnd_idxs[i,:]), max(rnd_idxs[i,:]))



    np.savetxt(fname=os.path.join(OUTPUT_DIR, 'start_indices_4k.csv'), X=np.transpose(rnd_idxs), fmt='%d', delimiter=',')
build_rnd_idxs()