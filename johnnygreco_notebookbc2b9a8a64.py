import numpy as np 

import pandas as pd
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
test.columns.tolist()
test.info()