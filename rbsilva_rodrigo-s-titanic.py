import numpy as np

import pandas as pd



train_df = pd.read_csv('../input/train.csv', header=0)

test_df = pd.read_csv('../input/test.csv', header=0)



train_df.info()