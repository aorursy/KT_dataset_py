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



from tqdm import tqdm



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torch.optim as optim





from fastai.basics import *

from fastai.callbacks import * 





class toydataset (Dataset):

    def __init__(self):

        

        

        self.dataset_size = 10

        self.data = torch.randn(self.dataset_size,150,12)

        

        self.targets= np.array([ 5.2743,  5.6834,  0.4407,  4.3582,  9.1221,  0.3569,  2.3914, 12.2206,

          3.9258,  8.0923],dtype='float32')



    def __len__(self):

        return self.dataset_size

             

    def __getitem__(self, idx):

        

        

        return self.data[idx,:,:] , self.targets[idx]

        
trn_ds = toydataset()

class RNN(nn.Module):

    def __init__(self, input_size = 12, hidden_size=48, num_layers=1, bidirectional=False):

        

        super().__init__()

        

        self.input_size = input_size

        self.hidden_size = hidden_size

        

        self.bidirectional,self.num_layers = bidirectional,num_layers

        if bidirectional: self.num_directions = 2

        else: self.num_directions = 1

                                

        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=self.bidirectional, batch_first=True, num_layers = num_layers)

       

        self.final_layers = nn.Sequential(

            

            nn.Linear(self.num_directions * hidden_size,10),    

            

            nn.ReLU(),        

            nn.Linear(10,1),    

        )

        

    def forward(self,input_seq):

    

      

        output, h_n = self.rnn(input_seq)        

        

        output = output[:,-1,:]

        

        output = self.final_layers(output)

        

                

        return output
# CUDA for PyTorch

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
net = RNN()

criterion =  nn.L1Loss()



databunch = DataBunch.create(train_ds= trn_ds,valid_ds = trn_ds, device=device, bs=10)



learn = Learner(databunch,net,callback_fns=[ShowGraph], loss_func = criterion, wd =0)

# Pre training preds+targets

learn.get_preds(ds_type=DatasetType.Train)
lr=1e-3

learn.fit(250,lr)
# Post training preds+targets

learn.get_preds(ds_type=DatasetType.Train)
x,y = learn.get_preds(ds_type=DatasetType.Train)

criterion(x,y)