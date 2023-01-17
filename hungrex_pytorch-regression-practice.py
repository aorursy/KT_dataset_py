# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch 
import torch.nn.functional as F

from tqdm import tqdm
class Net(torch.nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(Net, self).__init__() 
        self.h1 = torch.nn.Linear(n_in, n_hid)
        self.h2 = torch.nn.Linear(n_hid, n_hid)
        self.predict = torch.nn.Linear(n_hid, n_out)
    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        return self.predict(x)
    

loss_F = torch.nn.MSELoss()
model = Net(n_in = 2, n_hid = 6, n_out= 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(model)
x = torch.arange(10).reshape(5, 2)
y = torch.arange(5)
for epoch in range(300):
    for i in range(5):
#         print(x[i])
        predict = model(x[i].float())
#         print(y[i])
        loss = loss_F(predict, y[i].float())
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()  
        optimizer.zero_grad()   # clear gradients for next train
#         print(loss)
        
test = torch.tensor([10,11])
print(test)
print(model(test.float()))

import torch
print(torch.__version__)
