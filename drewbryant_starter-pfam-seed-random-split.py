import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn
data_partitions_dirpath = '../input/random_split/random_split'

print('Available dataset partitions: ', os.listdir(data_partitions_dirpath))
def read_all_shards(partition='dev', data_dir=data_partitions_dirpath):

    shards = []

    for fn in os.listdir(os.path.join(data_dir, partition)):

        with open(os.path.join(data_dir, partition, fn)) as f:

            shards.append(pd.read_csv(f, index_col=None))

    return pd.concat(shards)



test = read_all_shards('test')

dev = read_all_shards('dev')

train = read_all_shards('train')



partitions = {'test': test, 'dev': dev, 'train': train}

for name, df in partitions.items():

    print('Dataset partition "%s" has %d sequences' % (name, len(df)))
dev.head()
dev.groupby('family_id').size().sort_values(ascending=False).head(10)
for name, partition in partitions.items():

    partition.groupby('family_id').size().hist(bins=50)

    plt.title('Distribution of family sizes for %s' % name)

    plt.ylabel('# Families')

    plt.xlabel('Family size')

    plt.show()
dev['alignment_length'] = dev.aligned_sequence.str.len()

dev.alignment_length.hist(bins=30)

plt.title('Distribution of alignment lengths')

plt.xlabel('Alignment length')

plt.ylabel('Number of sequences')
family_lengths = (dev[['family_id', 'alignment_length']]

                  .drop_duplicates()

                  .sort_values(by='alignment_length', ascending=False))



family_lengths.head(5)
family_lengths.tail(5)