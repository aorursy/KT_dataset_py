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
import torch
from torch.utils.data import DataLoader, Dataset
class my_dataset(Dataset):
   
    def __init__(self, res):        
        self.res = res
    
    def __len__(self):
        return N_ELEM
    
    def __getitem__(self, idx):
        return torch.rand(3,self.res, self.res)
N_ELEM = 64
BATCH_SIZE = 16
dataset_loader = DataLoader(my_dataset(256), 
                            batch_size=BATCH_SIZE,
                            shuffle=False, 
                            num_workers=2)
%%time
for i_batch, sample in enumerate(dataset_loader):
    sample = sample*2
dataset_loader = DataLoader(my_dataset(756), 
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=2)
%%time
for i_batch, sample in enumerate(dataset_loader):
    sample = sample*2
!df -h
