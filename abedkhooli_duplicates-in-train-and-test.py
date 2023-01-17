import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# read data sets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print("Train shape:", train.shape)

print("Test shape:", test.shape)



# columns to determine duplicates (ID and y excluded)

cols = [key for key in train.keys()][2:]

dups_train = train.duplicated(cols, keep=False)



# sort to have a look

dtrain = train[dups_train].sort_values(cols)

print("Train dups shape:", dtrain.shape)



# do same for test data

dups_test =test.duplicated(cols, keep=False)

dtest = test[dups_test].sort_values(cols)

print("Test dups shape:", dtest.shape)



# find common duplicate rows in train and test 

common = pd.merge(dtrain, dtest, on=cols, how='inner')

print("Common dups shape:", common.shape)
# common rows in train and test

comm_all = pd.merge(train, test, on=cols, how='inner')

comm_all.shape
#possibly missing features

columns = train.keys()

for i in range(386):

    if not ("X"+str(i)) in cols:

        print("X"+str(i))