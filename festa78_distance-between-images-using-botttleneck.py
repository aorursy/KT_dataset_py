"""Compute distance between images using botttleneck features.
Use bottleneck features extracted by:
https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1
ref. https://www.kaggle.com/festa78/extract-bottleneck-features-using-tensorflow-hub
"""

from threading import Thread
import os
import time

import glob
import numpy as np
import pandas as pd
import tqdm
BOTTLENECK_DIR = '../input/google-landmark-sample-bottleneck'
TEST_DIR = '../input/google-landmark-sample-bottleneck'
NUM_FEATURES = 2048
# Read image list.
bottleneck_list = glob.glob(os.path.join(BOTTLENECK_DIR, '*.txt'))
bottleneck_base = [os.path.basename(filename) for filename in bottleneck_list]
test_list = glob.glob(os.path.join(TEST_DIR, '*.txt'))
test_base = [os.path.basename(filename) for filename in test_list]
print(bottleneck_list[:3])
print(bottleneck_base[:3])
# Compute distance
DATASIZE_WORKER = 2
THREADS_MAX = 100
lt = np.hstack([np.arange(0, len(test_list), DATASIZE_WORKER), len(test_list)])
dist_all = pd.DataFrame(np.zeros([len(test_list),  len(bottleneck_list)]), index=test_base, columns=bottleneck_base)

# process per data chunk.
def corr_worker(idx_from, idx_to):
    dist_worker = pd.DataFrame(np.zeros([idx_to - idx_from, len(bottleneck_list)]),
                               index=test_base[idx_from:idx_to],
                               columns=bottleneck_base)
    for i in range(idx_from, idx_to):
        with open(test_list[i], 'r') as f:
            line = f.readline()
            test_i = np.array([float(item) for item in line.split(',')])
        for j, filepath in enumerate(bottleneck_list):
            with open(filepath, 'r') as f:
                line = f.readline()
                bottleneck_j = np.array([float(item) for item in line.split(',')])
            dist_all.iloc[i, j] = np.linalg.norm(test_i - bottleneck_j)
            dist_worker.iloc[i - idx_from, j] = np.linalg.norm(test_i - bottleneck_j)
    print('test_{:09d}_{:09d}.csv'.format(idx_from, idx_to))
    # dist_worker.to_csv(os.path.join(OUT_DIR, 'test_{:09d}_{:09d}.csv'.format(idx_from, idx_to)))

t1 = time.time()
threads = []
for i in tqdm.tqdm(range(len(lt) - 1), total=len(lt) - 1):
    thread = Thread(target=corr_worker, args=(lt[i], lt[i + 1]))
    thread.start()
    threads.append(thread)
    if len(threads) > THREADS_MAX:
        for thread in threads:
            thread.join()
        threads = []
for thread in threads:
    thread.join()

t2 = time.time()
 
print(dist_all)