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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vision
import torchvision.datasets as DS
import torchvision.transforms as transforms
T=transforms.Compose([
    transforms.Resize([28,28]),
    transforms.ToTensor()
])
path="/kaggle/input/chest-xray-pneumonia/chest_xray/train/"
ds=DS.ImageFolder(root=path,transform=T)
ds
ds[0][0].shape
# 3*1858*2090
input_size=3*28*28
output_size=2
hidden_size=32
batch_size=80
dl=DataLoader(ds,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=2)
for i,j in dl:
    print(i.shape)
    break
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device)

    
def accuracy(op,label):
    _,pred=torch.max(op,dim=1)
    return torch.tensor(torch.sum(pred==label).item()/len(pred))



class Model(nn.Module):
    def __init__(self,ip_size,hidden_size,op_size):
        super().__init__()
        self.linear1=nn.Linear(ip_size,hidden_size)
        self.linear2=nn.Linear(hidden_size,op_size)
    
    def forward(self,xb):
        xb=xb.view(xb.size(0),-1)
        out=self.linear1(xb)
        out=F.relu(out)
        out=self.linear2(out)
        return out
    
    
        
        
class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl=dl
        self.device=device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b,self.device)
    def __len__(self):
        return len(self.dl)
        
device=get_device()
device
dl=DeviceDataLoader(dl,device)
for i,j in dl:
    img=i
    l=j
    break
model=Model(input_size,hidden_size,output_size)
to_device(model,device)
pred=model(img)
loss=F.cross_entropy(pred,l)
print(loss)
acc=accuracy(pred,l)
print(acc)
def fit(epoch,lr,model,loss,train):
    optim=torch.optim.SGD(model.parameters(),lr=lr)
    for i in range(epoch):
        for i,j in train:
            pred=model(i)
            ls=loss(pred,j)
            ls.backward()
            optim.step()
            optim.zero_grad()
            acc=accuracy(pred,j)
            return {"loss":ls,"acc":acc}   
loss=F.cross_entropy
fit(5,0.05,model,loss,dl)
val=DS.ImageFolder(root="/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val/",transform=T)
len(val)
def test(model,loss,val):
    arr=[]
    for i,j in val:
        pred=model(i)
        ls=loss(pred,j)
        acc=accuracy(pred,j)
#         print({"loss":ls,"acc":acc})
        arr.append({"loss":ls,"acc":acc})
    return arr
vl=DataLoader(val,batch_size=4,shuffle=True)
vl=DeviceDataLoader(vl,device)
lst=test(model,loss,vl)
torch.stack([x["acc"] for x in lst]).mean()
val=DS.ImageFolder(root="/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/",transform=T)
print(len(ds))
tl=DataLoader(val,batch_size=4,shuffle=True)
tl=DeviceDataLoader(tl,device)
lst1=test(model,loss,tl)
torch.stack([x["acc"] for x in lst1]).mean()
