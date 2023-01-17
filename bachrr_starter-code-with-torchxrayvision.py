!pip install -qU torchxrayvision
from glob import glob

import matplotlib.pyplot as plt

import numpy as np

import torch  

import torchvision

import torchxrayvision as xrv

import pylab

import torch.optim as optim

import torch.nn as nn

from pathlib import Path

from tqdm import tqdm
PATH = Path('../input/covid-chest-xray')
transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

dataset = xrv.datasets.COVID19_Dataset(imgpath=PATH/'images',csvpath=PATH/'metadata.csv', transform=transform)
print(dataset)
len_dataset=len(dataset)

n_train=int(0.9*len_dataset)

n_test=int(0.1*len_dataset)+1

print(f'Total samples: {len_dataset}, train size size: {n_train}, test set size: {n_test}')
train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train,n_test])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4,shuffle=True, num_workers=4)

test_dl = torch.utils.data.DataLoader(test_ds, batch_size=4,shuffle=True, num_workers=1)
model = xrv.models.DenseNet(num_classes=2).cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
features = 'PA' # 

target = 'lab'  #
train_loss_history, test_loss_history = [], []

for epoch in tqdm(range(10)):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(train_dl, 0):

        inputs=data[features].cuda()

        labels=data[target].long().cuda()

        labels=labels[:,2]

       

        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

    test_loss=0.0



    for i, data in enumerate(test_dl, 0):

        inputs=data[features].cuda()

        labels=data[target].long().cuda()

        labels=labels[:,2]



        # forward + backward + optimize

        outputs = model(inputs)

        loss = criterion(outputs, labels)



        # print statistics

        test_loss += loss.item()

        

    train_loss_history.append(running_loss)

    test_loss_history.append(test_loss)
plt.plot(train_loss_history, label='trainig loss')

plt.plot(test_loss_history, label='testing loss')

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()