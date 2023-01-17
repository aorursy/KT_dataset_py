import numpy as np

import pandas as pd
members = pd.read_csv('../input/members.csv')
members.shape
members.isnull().sum()
print("{0:.2f}% data for gender is missing".format(3354778 / 5116194 * 100))
transactions = pd.read_csv('../input/transactions.csv')
transactions.shape
transactions.isnull().sum()
train = pd.read_csv('../input/train.csv')
train.shape
train.isnull().sum()
''''train = pd.read_csv('../input/train.csv')

sample = pd.read_csv('../input/sample_submission_zero.csv')

transactions = pd.read_csv('../input/transactions.csv')

'''