import torch

import torchvision

from torchvision import transforms, datasets



import matplotlib.pyplot as plt

import numpy as np
#both training and testing dataset is coverted to tensor



train = datasets.MNIST('', train=True, 

                      download=True, transform=transforms.Compose(

                      [transforms.ToTensor()]))



test = datasets.MNIST('', train=False, 

                      download=True, transform=transforms.Compose(

                      [transforms.ToTensor()]))
trainLoader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

testLoader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)
next(iter(trainLoader))
#unpacking

X, y = next(iter(trainLoader))
plt.imshow(X[1].reshape(28,-1), cmap='gray')
print(y[1])
total = 0

counter = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}



for i in trainLoader:

    Xs, ys = i

    for y in ys:

        counter[int(y)] += 1

        total += 1

        

print(counter)

print(total)
import torch.nn as nn

import torch.nn.functional as F
class LinNets(nn.Module):

    def __init__(self):

        super(LinNets, self).__init__()

        self.fc1 = nn.Linear(1*28*28, 32)

        self.fc2 = nn.Linear(32, 32)

        self.fc3 = nn.Linear(32, 16)

        self.fc4 = nn.Linear(16, 10)

    

    #play with activation function

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        

        #research about softmax function

        return F.softmax(x, dim=1)

    

model = LinNets()

model
# 1, 784 | 784,32 | 32, 32 | 32,16 | 16,8 | 8,10 => 1,10



randomX = torch.randn(28, 28)

randomX = randomX.reshape(1,-1)



import pandas as pd
y = model(randomX) #training



y = pd.DataFrame(y.detach().numpy(), columns=[0,1,2,3,4,5,6,7,8,9])

y #prediction
columns=[0,1,2,3,4,5,6,7,8,9]



plt.bar(columns, y.T[0])
for ind, param in enumerate(model.parameters()):

    print(f'layer{ind}: ', param)
import time
#training in GPU



start = time.time()

device = torch.device('cuda')



model = model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=0.01)



losses = []



EPOCHS = 5



for i in range(EPOCHS):

    for X,y in trainLoader:

        X = X.to(device)

        y = y.to(device)

        

        y_pred = model(X.reshape(-1,28*28))

        loss = F.cross_entropy(y_pred,y)

        

        #gradient descent or back propagation

        loss.backward()

        optim.step()

    

    print(loss)



end = time.time()

print('total Time: ', end-start)
#training in CPU



start = time.time()

device = torch.device('cpu')



model = model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=0.01)



EPOCHS = 5



for i in range(EPOCHS):

    for X,y in trainLoader:

        X = X.to(device)

        y = y.to(device)

        

        y_pred = model(X.reshape(-1,28*28))

        loss = F.cross_entropy(y_pred,y)

        

        #gradient descent or back propagation

        loss.backward()

        optim.step()

    

    print(loss)



end = time.time()

print('total Time: ', end-start)