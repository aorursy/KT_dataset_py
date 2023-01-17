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
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
%matplotlib inline
data=pd.read_csv('../input/sp_d.csv')

data['Date']=pd.to_datetime(data['Date'])
plt.plot(data['Date'],data['Close'])
data['V']=(data['Close'].shift(1)-data['Close'])/data['Close'].shift(1)
data2=data[['Date','V']]
plt.plot(data2['Date'],data2['V'])
df=data2.dropna()
df.head()
dataset=df['V'].values

dataset=dataset.astype('float32')

max_v=np.max(dataset)

min_v=np.min(dataset)

dmm=max_v-min_v

dataset=list(map(lambda x: x/dmm,dataset))
x=np.array(dataset[:-1])

y=np.array(dataset[1:])
x.shape
y.shape
train_size=int(len(x)*0.8)

test_size=len(x)-train_size

train_x=x[:train_size]

train_y=y[:train_size]

test_x=x[train_size:]

test_y=y[train_size:]
train_x=train_x.reshape(-1,1,1)

train_y=train_y.reshape(-1,1,1)

test_x=test_x.reshape(-1,1,2)


train_x=torch.from_numpy(train_x)

train_y=torch.from_numpy(train_y)

test_x=torch.from_numpy(test_x)
train_x.shape
Variable(train_x).shape
class RNN(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size=1,num_layers=2):
        
        super(RNN,self).__init__()
        
        self.rnn=nn.LSTM(input_size,hidden_size,num_layers)
        
        self.fc=nn.Linear(hidden_size,output_size)
        
    def forward(self,x):
        
        x,_=self.rnn(x) #(seq,batch,hidden)
        
        s,b,h=x.shape
        
        x=x.view(s*b,h)
        
        x=self.fc(x)
        
        x=x.view(s,b,-1)
        
        return x        
net=RNN(1,4)

criterion=nn.MSELoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.01)
for e in range(4):
    
    var_x=Variable(train_x)
    
    var_y=Variable(train_y)
    
    out=net(var_x)
    
    loss=criterion(out,var_y)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    print('Epoch: {}, Loss: {:.5f}'.format(e+1,loss.data[0]))
net=net.eval()
x=x.reshape(-1,1,1)

x=torch.from_numpy(x)

var_data=Variable(x)

pred_test=net(var_data)
pred_test=pred_test.view(-1).data.numpy()
plt.plot(pred_test,'r',label='prediction')

plt.plot(dataset,'b',label='real')

plt.legend()
