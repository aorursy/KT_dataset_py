import pandas as pd

import numpy as np
train_orig = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int32, 'time_to_failure': np.float32})
train_orig['time_to_failure'].values.mean()
# The median value?

np.median(train_orig['time_to_failure'].values)
# The min value?

np.min(train_orig['time_to_failure'].values)
# The 1% value?

np.quantile(train_orig['time_to_failure'].values, 0.01)
import os

num_test_samples = len(os.listdir('../input/test/'))

print('There are',num_test_samples,'test samples')
0.13 * 2624
np.append(np.repeat(4.017, 341), np.repeat(5.926423, 2624-341)).mean()