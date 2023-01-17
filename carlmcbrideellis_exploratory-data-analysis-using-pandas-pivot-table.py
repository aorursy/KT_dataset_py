import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

# read the training data into a pandas DataFrame

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train_data
pivot_table = pd.pivot_table(train_data, index=['OverallQual'])

pd.set_option('display.max_columns', None)

# take a look

pivot_table
pivot_table = pd.pivot_table(train_data, index=['OverallQual'], values=['GrLivArea','SalePrice'])

pivot_table
pivot_table = pd.pivot_table(train_data, index=['OverallQual'], values=['SalePrice'])

# now plot

pivot_table.plot(kind='bar');