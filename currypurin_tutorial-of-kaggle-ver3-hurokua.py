import pandas as pd

import pandas_profiling as pdp
# データの読み込み

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
pdp.ProfileReport(train)
pdp.ProfileReport(test)
full = pd.concat([train, test], axis=0, sort=False)

full.shape
pdp.ProfileReport(full)