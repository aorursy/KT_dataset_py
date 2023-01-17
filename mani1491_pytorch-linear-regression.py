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

from torch.autograd import Variable 

# Building X

# n- number of rows, m- number of features

n,m=500,15

x=torch.randn(n,m)



# Building Y

y=torch.randn(n,1)
# Model Class



class linearreg(torch.nn.Module):

    

    def __init__(self,ncols_x,ncols_y):

        

        super(linearreg,self).__init__()

        self.linear=torch.nn.Linear(ncols_x,ncols_y)

        

    def forward(self,x):

        

        y_hat=self.linear(x)

        return y_hat

                

        

        

        

        
model=linearreg(m,1)
# Loss Criterion

criterion = torch.nn.MSELoss()# Mean Squared Loss

optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
# Training

for epoch in range(100):

    

    #print(list(model.parameters())[0].grad)

    y_hat=model(x)

    



    loss=criterion(y_hat,y,)

    print('Epoch: '+ str(epoch) , loss.item())



    # Backprop and update weights

    optimizer.zero_grad() # set the gradients to zero to avoid gradient accumulation from previous steps

    loss.backward() # back propagation

    optimizer.step() # update the weights

    





    



# Predicting



test_x=torch.randn(m)

y_pred = model(test_x)

y_pred.item()