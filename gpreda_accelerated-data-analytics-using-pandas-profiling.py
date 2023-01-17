import pandas as pd

import pandas_profiling
data_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
pandas_profiling.ProfileReport(data_df)