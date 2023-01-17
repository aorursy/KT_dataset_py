# Importing Libraries



import os

from os import listdir as ld





list(ld("../input/osic-pulmonary-fibrosis-progression"))
# Importing libraries



import pandas as pd



train_dataset=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_dataset=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
train_dataset.head()
test_dataset.head()
print(" train_dataset shape is :",train_dataset.shape)

print(" test_dataset shape is :",test_dataset.shape)
train_dataset.dtypes
train_dataset.info()
train_dataset.isna().sum()
test_dataset.info()
test_dataset.isna().sum()