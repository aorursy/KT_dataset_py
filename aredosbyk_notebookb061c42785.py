import pandas as pd

import numpy as np

from tqdm import tqdm
train_X = pd.read_csv('../input/lish-moa/train_features.csv', index_col='sig_id')

test_Y = pd.read_csv('../input/lish-moa/sample_submission.csv', index_col='sig_id')

train_Y = pd.read_csv('../input/lish-moa/train_targets_scored.csv', index_col='sig_id', dtype={f: test_Y.dtypes[f] for f in test_Y})

test_X = pd.read_csv('../input/lish-moa/test_features.csv', index_col='sig_id')
train_X
train_Y
test_X
test_Y
train_Y.sum(1).value_counts()
for f in test_Y:

    test_Y[f] = train_Y[f].mean()
test_Y.to_csv('submission.csv')