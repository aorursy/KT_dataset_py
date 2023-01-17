# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# more information: http://github.com/h2oai/datatable

!pip install datatable
import datatable as dt

import time

t1 = time.time()

folder_path = '../input/'



# parse all files

train_identity = dt.fread(f'{folder_path}train_identity.csv')

test_identity = dt.fread(f'{folder_path}test_identity.csv')

train_transaction = dt.fread(f'{folder_path}train_transaction.csv')

test_transaction = dt.fread(f'{folder_path}test_transaction.csv')

t2 = time.time()

print("Time to parse: %f" % (t2 - t1))



# join frames

train_identity.key = 'TransactionID'

test_identity.key = 'TransactionID'

train = train_transaction[:, :, dt.join(train_identity)]

test = test_transaction[:, :, dt.join(test_identity)]

t3 = time.time()

print("Time to join: %f" % (t3 - t2))



# save as .csv

train.to_csv("train.csv", _strategy="write")  # "write" strategy saves memory, instead of default memory mapping

test.to_csv("test.csv", _strategy="write")

t4 = time.time()

print("Time to save as .csv: %f" % (t4 - t3))



# turn datatable into Pandas frames

train_df = train.to_pandas()

test_df = test.to_pandas()

t5 = time.time()

print("Time to convert: %f" % (t5 - t4))



print("Total time: %f" % (t5 - t1))

print(train_df.shape)

print(test_df.shape)