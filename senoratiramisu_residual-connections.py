import numpy as np 

import pandas as pd 

import torch

from torch import nn

from torch.optim import Adam

import torchvision

import torchvision.transforms as transforms
transform = transforms.Compose(

    [transforms.ToTensor(),

     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



bs = 128



trainset = torchvision.datasets.CIFAR10(root='./data', train=True,

                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,

                                          shuffle=True, num_workers=4)



testset = torchvision.datasets.CIFAR10(root='./data', train=False,

                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=bs,

                                         shuffle=False, num_workers=4)



classes = ('plane', 'car', 'bird', 'cat',

           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class Block(nn.Module):

    def __init__(self,c_in,c_out,fs,p=0,rl = True):

        super().__init__()

        self.c_in, self.c_out, self.fs, self.rl = c_in, c_out, fs, rl

        self.conv = nn.Conv2d(self.c_in,self.c_out,self.fs,padding=p)

        self.norm, self.relu = nn.BatchNorm2d(self.c_out), nn.ReLU()

        

    def forward(self,x): 

        return self.relu(self.norm(self.conv(x))) if self.rl else self.norm(self.conv(x))
class ResBlock(nn.Module):

    def __init__(self,nc,fs):

        super().__init__()

        self.nc, self.fs = nc, fs

        self.a = Block(self.nc,self.nc,self.fs,p=1)

        self.b = Block(self.nc,self.nc,self.fs,rl=None,p=1)

        self.relu = nn.ReLU()

        

    def forward(self,x):

        y = self.a(self.b(x))

        return self.relu(x+y)
class AltBlock(nn.Module):

    def __init__(self,nc,fs1,fs2):

        super().__init__()

        self.nc, self.fs1, self.fs2 = nc, fs1, fs2

        self.a = Block(self.nc,self.nc,self.fs1)

        self.b = Block(self.nc,self.nc,self.fs1, rl = None)

        self.c = Block(self.nc,self.nc,self.fs2, rl = None)

        self.relu = nn.ReLU()

        

    def forward(self,x):

        y = self.a(self.b(x))

        z = self.c(x)

        return self.relu(y+z)
net0 = nn.Sequential(Block(3,6,5),

                     Block(6,16,3),

                     Block(16,16,3),

                     Block(16,16,3),

                     Block(16,32,3),

                     Block(32,32,3),

                     Block(32,32,3),

                     Block(32,64,3),

                     Block(64,64,3),

                     Block(64,64,3),

                     nn.AdaptiveAvgPool2d(1),

                     nn.Flatten(),

                     nn.Linear(64,10)

                     )
net1 = nn.Sequential(Block(3,6,5),

                     Block(6,16,3),

                     Block(16,16,5),

                     Block(16,32,3),

                     Block(32,32,5),

                     Block(32,64,3),

                     Block(64,64,5),

                     nn.AdaptiveAvgPool2d(1),

                     nn.Flatten(),

                     nn.Linear(64,10)

                     )
net2 = nn.Sequential(Block(3,6,5),

                     Block(6,16,3),

                     ResBlock(16,3),

                     Block(16,32,3),

                     ResBlock(32,3),

                     Block(32,64,3),

                     ResBlock(64,3),

                     nn.AdaptiveAvgPool2d(1),

                     nn.Flatten(),

                     nn.Linear(64,10)

                     )
net3 = nn.Sequential(Block(3,6,5),

                     Block(6,16,3),

                     AltBlock(16,3,5),

                     Block(16,32,3),

                     AltBlock(32,3,5),

                     Block(32,64,3),

                     AltBlock(64,3,5),

                     nn.AdaptiveAvgPool2d(1),

                     nn.Flatten(),

                     nn.Linear(64,10))
models = [net0, net1, net2, net3]
def train_loop(model, optimizer, criterion, epochs):

    

    metrics = []

    

    for _ in range(epochs):

        current = 0

        model.train()

        for img, lab in trainloader:

            

            optimizer.zero_grad()

            out = model(img.float().cuda())

            loss = criterion(out, lab.cuda())

            loss.backward()

            optimizer.step()

            current += loss.item()

            

        train_loss = current / len(trainloader)

            

        with torch.no_grad():

            current, acc = 0, 0

            model.eval()

            for img, lab in testloader:

                out = model(img.float().cuda())

                loss = criterion(out, lab.cuda())

                current += loss.item() 

                _, pred = nn.Softmax(-1)(out).max(-1)

                acc += (pred == lab.cuda()).sum().item()

            

            valid_loss = current / len(testloader)

            accuracy = 100 * acc / len(testset)

            

        metrics.append([train_loss,valid_loss,accuracy])

        

    return np.array(metrics)
def get_results(models,epochs, lr):

    

    tuples = list(zip(*[3*['net0'] + 3*['net1'] + 3*['net2'] + 3*['net3'],4*['train_loss','valid_loss','accuracy']]))

    index = pd.MultiIndex.from_tuples(tuples, names=['model', 'metric'])

    results = pd.DataFrame(index = range(epochs),columns = index)

    

    for i,model in enumerate(models): 

        results[f'net{i}'] = train_loop(

                                        model = model.cuda(),

                                        optimizer = Adam(model.parameters(),lr=lr),

                                        criterion = nn.CrossEntropyLoss(),

                                        epochs = epochs

                                       )



    return results
get_results(models = models, epochs = 7, lr = 2e-3)