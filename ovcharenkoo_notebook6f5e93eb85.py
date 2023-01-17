%matplotlib inline

import numpy as np

import pandas as pd



train = pd.read_csv('../input/train.csv', header = 0)

test = pd.read_csv('../input/test.csv', header = 0)

full = [train, test]
print(train.info())