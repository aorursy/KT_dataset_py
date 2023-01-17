# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import torch

import torch.nn as nn

import torchvision.transforms as transforms

import torchvision.datasets as dsets

import matplotlib.pyplot as plt

import sklearn

import sklearn.preprocessing
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

min_max_scaler = sklearn.preprocessing.MinMaxScaler()
allFileNames = os.listdir('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/')

np.random.shuffle(allFileNames)

train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),

                                                          [int(len(allFileNames)*0.75), int(len(allFileNames)*0.80)])

train_FileNames = ['../input/price-volume-data-for-all-us-stocks-etfs/Stocks/'+ name for name in train_FileNames.tolist()]

val_FileNames = ['../input/price-volume-data-for-all-us-stocks-etfs/Stocks/'+ name for name in val_FileNames.tolist()]

test_FileNames = ['../input/price-volume-data-for-all-us-stocks-etfs/Stocks/' + name for name in test_FileNames.tolist()]
print(len(train_FileNames)/float(len(allFileNames)))

print(len(val_FileNames)/float(len(allFileNames)))

print(len(test_FileNames)/float(len(allFileNames)))

train_FileNames[0]
import matplotlib.pyplot as plt

plt.close('all')
exploredf = pd.read_csv(train_FileNames[0], sep=",")
exploredf['Open'].plot()
exploredf[["Open", "High","Low","Close","Volume"]] = min_max_scaler.fit_transform(exploredf[["Open", "High","Low","Close","Volume"]])
exploredf=exploredf.drop(['Date', 'OpenInt'],axis=1)
exploredf
batch_size = 1

seq_len = 20

input_dim = 5

hidden_dim = 100

n_layers = 10

#lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

#inp = torch.randn(batch_size, seq_len, input_dim)

#hidden_state = torch.randn(n_layers, batch_size, hidden_dim)

#cell_state = torch.randn(n_layers, batch_size, hidden_dim)

#hidden = (hidden_state, cell_state)