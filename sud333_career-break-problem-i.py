import pandas as pd

import numpy as np
train = pd.read_csv('../input/train.csv')
train.head(3)
ser = train.count()

ser[ser < 30471]/30471