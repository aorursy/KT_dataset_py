import numpy as np

import torch as tc

import pandas as pd



import scipy.stats as stats

import statsmodels.api as sm



from tqdm import tqdm as tqdm



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
features = pd.read_csv('../input/lish-moa/train_features.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')

target = pd.read_csv('../input/lish-moa/train_targets_scored.csv')



assert features.sig_id.duplicated().sum() == 0

assert test_features.sig_id.duplicated().sum() == 0

assert target.sig_id.duplicated().sum() == 0



features = features.set_index('sig_id').sort_index()

test_features = test_features.set_index('sig_id').sort_index()

target = target.set_index('sig_id').sort_index()
features.cp_type.unique()
test_target = np.zeros((len(test_features), len(target.columns)))

test_target[test_features.cp_type == 'trt_cp'] = target[features.cp_type == 'trt_cp'].mean().values

test_target = pd.DataFrame(data=test_target, index=test_features.index, columns=target.columns)

test_target.head()
test_target.to_csv('submission.csv')