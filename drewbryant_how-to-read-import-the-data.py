import os

import numpy as np

import pandas as pd



data_partitions_dirpath = '../input/random_split'

print('Available dataset partitions: ', os.listdir(data_partitions_dirpath))
def read_all_shards(partition='dev', data_dir='../input/random_split'):

    shards = []

    for fn in os.listdir(os.path.join(data_dir, partition)):

        with open(os.path.join(data_dir, partition, fn)) as f:

            shards.append(pd.read_csv(f, index_col=None))

    return pd.concat(shards)



test = read_all_shards('test')

dev = read_all_shards('dev')

train = read_all_shards('train')



test.shape, dev.shape, train.shape