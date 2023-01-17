import pandas as pd

import matplotlib

import warnings

warnings.filterwarnings('ignore')

import pandas_profiling as pdp
# データの読み込み

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
pdp.ProfileReport(train_df)
profile
pdp.ProfileReport(test_df)