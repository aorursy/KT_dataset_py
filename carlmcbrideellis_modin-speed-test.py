!pip install modin[ray]

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time
start_time  = time.time()

train_data  = pd.read_csv('../input/santander-customer-satisfaction/train.csv')

test_data   = pd.read_csv('../input/santander-customer-satisfaction/test.csv')

finish_time = time.time()

no_modin_time = (finish_time - start_time)

print("Dataframes loaded: time taken %.2f" % no_modin_time +" seconds")
import modin.pandas as pd

start_time  = time.time()

train_data_again  = pd.read_csv('../input/santander-customer-satisfaction/train.csv')

test_data_again   = pd.read_csv('../input/santander-customer-satisfaction/test.csv')

finish_time = time.time()

with_modin_time = (finish_time - start_time)

print("Dataframes loaded: time taken %.2f" % with_modin_time +" seconds")