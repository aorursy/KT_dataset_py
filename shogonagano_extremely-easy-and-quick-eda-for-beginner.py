import pandas as pd

import pandas_profiling as pdp

train = pd.read_csv('../input/train/train.csv')

test = pd.read_csv('../input/test/test.csv')

pdp.ProfileReport(train)

pdp.ProfileReport(test)