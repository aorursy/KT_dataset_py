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
import torchvision
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
#These mean and std are the means and std of 3 channels(i.e RGB) of all the images 
mean=[0.49159136,0.48234546,0.44672027] 
std=[0.2383135,0.23491476,0.2526323 ]
#These lines will calculate the mean and std if you are intrested in the code
# pop_mean=[]
# pop_std=[]
# for data in dataloader:
#   img=data[0].numpy()
#   batch_mean=np.mean(img,axis=(0,2,3))
#   batch_std=np.std(img,axis=(0,2,3))
#   pop_mean.append(batch_mean)
#   pop_std.append(batch_std)

# pop_mean=np.array(pop_mean)
# pop_std=np.array(pop_std)

# pop_mean=pop_mean.mean(axis=0)
# pop_std=pop_std.mean(axis=0)

# print(pop_mean,pop_std)
train_transform=transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])
test_transform=transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

train_set=torchvision.datasets.CIFAR10(root='datasets/cifar10/train',download=True,train=True,transform=train_transform)
test_set=torchvision.datasets.CIFAR10(root='datasets/cifar10/train',train=False,transform=test_transform)

trainloader=DataLoader(train_set,batch_size=16,shuffle=True)
testloader=DataLoader(test_set,batch_size=16,shuffle=True)

print(train_set.classes)

in_size=3

hid1_size=16
hid2_size=32
out1_size=400
out2_size=10
kernel=5

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Sequential(
        nn.Conv2d(in_size,hid1_size,kernel),
        nn.BatchNorm2d(hid1_size),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer2=nn.Sequential(
            nn.Conv2d(hid1_size,hid2_size,kernel),
            nn.BatchNorm2d(hid2_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer3=nn.Sequential(
            nn.Linear(hid2_size*kernel*kernel,out1_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(out1_size,out2_size)
            
        )
    def forward(self,x):
        x=self.layer1(x)
        
        x=self.layer2(x)
#         print(x.shape)
        x=x.reshape(x.shape[0],-1)
#         print(x.shape)
        x=self.layer3(x)
        return F.log_softmax(x,dim=-1)

device=torch.device('cuda' if torch.cuda.is_available() else "cpu")
model=Net().to(device)

loss_fn=nn.NLLLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epochs=10
for epoch in range(epochs):
    for i,(img,target) in enumerate(trainloader):
        pred=model(img.to(device))
        loss=loss_fn(pred,target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%2000==0:
            
            print(loss)
            
model.eval()
correct=0
total=0
with torch.no_grad():
    for i,(img,target) in enumerate(testloader):
        output=model(img.to(device))
        pred=torch.argmax(output,dim=1)
        total+=img.shape[0]
        correct+=(pred==target.to(device)).cpu().sum().item()
print(f'Accuracy:{str(correct/total*100)}%')