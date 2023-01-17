import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

train = pd.read_csv("../input/en_train.csv")

test = pd.read_csv("../input/en_test.csv")
# check first 15 row in train set.

train.head(15)
# chekc first 15 row in test set.

test.head(15)
print("Total class: ", train['class'].unique().size)

train['class'].unique()