import pandas as pd

import numpy as np
## Importing the training dataset

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
print(train.info())
print(train.isnull().sum())