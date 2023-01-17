# loading dataset
from torchvision.datasets import CIFAR10
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
mnist = CIFAR10('data', train=True, download=True, transform=transform)
mnist
# create training and validation split
split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))
train , valid = index_list[:split], index_list[split:]
# create sample objects using Random Sampler
tr = SubsetRandomSampler(train)
vd = SubsetRandomSampler(valid)
# lets load data
tl = DataLoader(mnist, batch_size = 256, sampler = tr)
vl = DataLoader(mnist, batch_size = 256, sampler = vd)

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3, padding=1)
        self.conv2 = nn.Conv2d(16,32,3, padding=1)
        self.conv3 = nn.Conv2d(32,64,3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 1024)  # reshaping
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
model = Model()
from torch import optim
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1,11):
    t_loss, v_loss = [], []
    
    # training part
    
    for data, target in tl:
        optimizer.zero_grad()
        
        ## 1 forward propagation
        
        output = model(data)
        
        ## 2 loss function
        loss = loss_func(output, target)
        
        ## 3 backward propagation
        loss.backward()
        
        ## 4 wright optimization
        optimizer.step()
        
        t_loss.append(loss.item())
    
    # evaluation part
    model.eval()
    for data,target in vl:
        output = model(data)
        loss = loss_func(output, target)
        v_loss.append(loss.item())
        
    print('epoch',epoch, 'train_lo', np.mean(t_loss), 'val',np.mean(v_loss))
        
        
import matplotlib.pyplot as plt
# plt.plot(t_loss)
plt.plot(v_loss)