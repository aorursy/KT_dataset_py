import torch

import matplotlib.pyplot as plt

import numpy as np

import torch.nn.functional as F

from torch import nn

import pathlib

from torch.utils.data import DataLoader

from torchvision import *
device = torch.device("cpu")

transformtrain = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(9),

    transforms.ToTensor(),

    transforms.Normalize((.5, .5, .5), (.5, .5, .5))

])
transformtest = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize((.5, .5, .5), (.5, .5, .5))

])
trainds = datasets.ImageFolder('../input/fruits-360_dataset/fruits-360/Training/', transform=transformtrain)

testds = datasets.ImageFolder('../input/fruits-360_dataset/fruits-360/Test/', transform=transformtest)
trainloader = DataLoader(trainds, batch_size=64, shuffle=True)

testloader = DataLoader(testds, batch_size=32, shuffle=False)
root = pathlib.Path('../input/fruits-360_dataset/fruits-360/Training/')

classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
class ConvNet(nn.Module):



    def __init__(self, epoch, learningrate):

        super().__init__()

        self.epoch = epoch

        self.lr = learningrate

        self.conv1 = nn.Conv2d(3, 8, 5, 1, 2)

        self.conv2 = nn.Conv2d(8, 16, 5, 1, 2)

        self.fc1 = nn.Linear(8*8*16, 300)

        self.fc2 = nn.Linear(300, len(classes))



    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 8*8*16)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x
model = ConvNet(10, 0.0001).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), model.lr, weight_decay=0.0001)

trainlosses = []

testlosses = []
for e in range(model.epoch):

    trainloss = 0.0

    traintotal = 0

    trainsuccessful = 0

    for traininputs, trainlabels in trainloader:

        inputs, labels = traininputs.to(device), trainlabels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        _, trainpredict = torch.max(outputs.data, 1)

        loss.backward()

        optimizer.step()

        trainloss += loss.item()

        traintotal += labels.size(0)

        trainsuccessful += (trainpredict == labels).sum().item()



    else:

        testloss = 0.0

        testtotal = 0

        testsuccessful = 0

        with torch.no_grad():

            for testimages, testlabels in testloader:

                inputs_, labels_ = testimages.to(device), testlabels.to(device)

                predictions = model(inputs_)

                tloss = criterion(predictions, labels_)

                testloss += tloss.item()

                _, predict = torch.max(predictions.data, 1)

                testtotal += labels_.size(0)

                testsuccessful += (predict == labels_).sum().item()

        trainlosses.append(trainloss/len(trainloader))

        testlosses.append(testloss/len(testloader))

        print('Train Accuracy %{:.2f}'.format(100*trainsuccessful/traintotal))

        print('Test Accuracy %{:.2f}'.format(100*testsuccessful/testtotal))
plt.plot(trainlosses, label='Training loss', color='green')

plt.plot(testlosses, label='Validation loss', color='black')

plt.legend(frameon=False)

plt.show()