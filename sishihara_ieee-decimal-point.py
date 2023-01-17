import numpy as np

import pandas as pd

import math
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv',

                                usecols=['TransactionID', 'TransactionAmt'])
train_transaction.head()
train_transaction['TransactionAmtDecimalPoint'] = [math.modf(v)[0] for v in train_transaction['TransactionAmt']]
train_transaction.head()