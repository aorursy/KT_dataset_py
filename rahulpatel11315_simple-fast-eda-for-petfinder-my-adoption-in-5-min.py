import pandas as pd

import pandas_profiling as pp
train_data = pd.read_csv('../input/train/train.csv')
train_data.head()
pp.ProfileReport(train_data)
profile = pp.ProfileReport(train_data)

rejected_variables = profile.get_rejected_variables(threshold=0.9)

rejected_variables