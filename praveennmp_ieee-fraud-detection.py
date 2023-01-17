import numpy as np

import pandas as pd
test=pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
test

test.info()
test.describe()