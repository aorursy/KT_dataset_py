# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt
#  Store the label information as label_address_dict, key: label, value: addresses belonging to the label

domain = os.path.abspath('../input/bitcoin-partial-transaction-dataset/label')

label_address_dict = {}  # store the label addresses



for labeled_file_name in os.listdir('../input/bitcoin-partial-transaction-dataset/label'):

    filepath = os.path.join(domain, labeled_file_name)               # obtain the filename

    label = labeled_file_name.split('.')[0].split('-')[0]            # obtain the label

    if label not in label_address_dict:

        label_address_dict[label] = []

    with open(filepath, 'r') as f:

        lines = f.readlines()

        for line in lines:

            labeled_address = line.rstrip('\n')

            label_address_dict[label].append(labeled_address)



print('labels: ' + str(list(label_address_dict.keys())))

print('an example address belonging to BitcoinFog: ' + label_address_dict['BitcoinFog'][0])

nums = []

for key in label_address_dict.keys():

    nums.append(len(label_address_dict[key]))

plt.bar(list(label_address_dict.keys()), nums)

plt.ylabel('Number')

plt.title('Number of addresses belonging to the label')

for x,y in enumerate(nums):

    plt.text(x,y,'%s' %round(y,1),ha='center')

plt.show()
'''

Take dataset1_2014_11_1500000 as an example, 

we will display the files in dataset1_2014_11_1500000, 

which includes the information of the first 1,500,000 transaction in Nov. 2014.

'''

print('Description of blockhash.txt\n')

blockhash_df = pd.read_table('../input/bitcoin-partial-transaction-dataset/dataset1_2014_11_1500000/blockhash.txt',header=None, names=['blockID', 'bhash', 'btime', 'txs'],sep=' ',index_col=0)

print(blockhash_df.head(5))

print('Total rows: ' + str(len(blockhash_df)))
print('Description of txhash.txt\n')

txhash_df = pd.read_table('../input/bitcoin-partial-transaction-dataset/dataset1_2014_11_1500000/txhash.txt',header=None, names=['txID', 'txhash'],sep=' ',index_col=0)

print(txhash_df.head(5))

print('Total rows: ' + str(len(txhash_df)))
print('Description of addresses.txt\n')

addresse_df = pd.read_table('../input/bitcoin-partial-transaction-dataset/dataset1_2014_11_1500000/addresses.txt',header=None, names=['addrID', 'addr'],sep=' ',index_col=0)

print(addresse_df.head(5))

print('Total rows: ' + str(len(addresse_df)))
print('Description of tx.txt\n')

tx_df = pd.read_table('../input/bitcoin-partial-transaction-dataset/dataset1_2014_11_1500000/tx.txt',header=None, names=['txID', 'blockID', 'n_inputs', 'n_outputs', 'btime'],sep=' ',index_col=0)

print(tx_df.head(5))

print('Total rows: ' + str(len(tx_df)))
print('Description of txin.txt\n')

txin_df = pd.read_table('../input/bitcoin-partial-transaction-dataset/dataset1_2014_11_1500000/txin.txt',header=None, names=['txID', 'addrID', 'value'],sep=' ',index_col=0)

print(txin_df.head(5))

print('Total rows: ' + str(len(txin_df)))
print('Description of txout.txt\n')

txout_df = pd.read_table('../input/bitcoin-partial-transaction-dataset/dataset1_2014_11_1500000/txout.txt',header=None, names=['txID', 'addrID', 'value'],sep=' ',index_col=0)

print(txout_df.head(5))

print('Total rows: ' + str(len(txout_df)))