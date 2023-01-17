import numpy as np

import pandas as pd



pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16}, usecols=[0]).to_csv('train.gzip', index=False)