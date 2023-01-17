import pandas as pd

import numpy as np
train_input = pd.read_csv('../input/train.csv')

test_input = pd.read_csv('../input/test.csv')
print(train_input.shape)

train_input.head()
print(test_input.shape)

test_input.head()