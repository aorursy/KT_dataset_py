import pandas as pd

import warnings

warnings.filterwarnings('ignore')

import pandas_profiling as pdp
# データの読み込み

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
pdp.ProfileReport(train_df)
pdp.ProfileReport(test_df)
full = pd.concat([train_df, test_df], sort=False)
pdp.ProfileReport(full)