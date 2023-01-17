# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
length, num_pixels = df.shape
print(df.shape)
class dataset:
    def __init__(self,df, isTrain = True):
        self.df = df
        self.isTrain = isTrain
  
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if self.isTrain == True:
            label = np.zeros(10)
            label[self.df.iloc[idx,0]] = 1
            return self.df.iloc[idx,1:].to_numpy(), label
        else:
            return self.df.iloc[idx,:].to_numpy()
a = dataset(df)
x,y = a[0]
print(type(x))
print(type(y))
class Net(nn.Module):
    def __init__(self, Layers,p=0):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = F.relu(linear_transform(activation))
            else:
                activation = F.softmax(linear_transform(activation), dim = 0)
        return activation
def train(_model, criterion, train_loader, optimizer, epochs, scheduler):
    #ACC = []
    batch_size = train_loader.batch_size
    Total_loss = []
    for epoch in range(epochs):
        LOSS = 0
        iter = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = _model(x.type(torch.FloatTensor)).reshape(10,-1)
            #print(yhat.shape)
            y = y.type(torch.FloatTensor)
            y = y.reshape(10,-1)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS += torch.sum(torch.abs(loss)).item()/batch_size
            if iter%100==0:
                print('Loss iter '+ str(iter) + ':'+ str(round(torch.sum(torch.abs(loss)).item()/batch_size,5)) )
            iter+=1
        #ACC.append(accuracy(_model, data_set))
        Total_loss.append(LOSS)
        print("epoch: "+ str(epoch)+ "Loss = " + str(round(LOSS/int(length/batch_size),5)))
        scheduler.step()
    return Total_loss
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(dev)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers = [784,400,100,10]
model = Net(layers)
dset = dataset(df)
train_loader = DataLoader(dset,batch_size=32,shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.01)
LOSS = train(model, criterion, train_loader, optimizer, epochs=10, scheduler = scheduler)
plt.plot(LOSS)
plt.show()
t_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
t_length, t_num_pixels = t_df.shape
print(t_df.shape)
tset = dataset(t_df, False)
x = tset
print(type(x[0]))
model2 = model.eval()
out = []
#testval = x[190]
#a = testval.reshape(28,28)
with torch.no_grad():
    for i in x:
        yhat = model2(torch.FloatTensor(i))
        out.append(np.argmax(yhat.detach().numpy()))
#    print(np.argmax(yhat.detach().numpy()))
#    print(yhat)
#plt.imshow(a, interpolation='nearest')
#plt.show()
len(out)
id = list(range(28000))
id = np.add(id,1)
dic = {'ImageId': id, 'Label':out}
save_df = pd.DataFrame(dic)
save_df.to_csv("submit.csv",index=False)
save_df
