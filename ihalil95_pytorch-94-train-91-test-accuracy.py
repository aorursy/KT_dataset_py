import torch

import torch.nn as nn

from torchvision import *

from matplotlib import pyplot as plt

from torch.utils.data import DataLoader

import torch.nn.functional as F

import pathlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformtrain = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
transformtest = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
datasettrain = datasets.ImageFolder('../input/data/synthetic_digits/imgs_train', transform=transformtrain)

datasettest = datasets.ImageFolder('../input/data/synthetic_digits/imgs_valid', transform=transformtest)
loadertrain = DataLoader(datasettrain, batch_size=128, shuffle=True)

loadertest = DataLoader(datasettest, batch_size=32, shuffle=False)
root = pathlib.Path('../input/data/synthetic_digits/imgs_valid')

classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

classes
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
model = ConvNet(40, 0.0002).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), model.lr, weight_decay=0.0001)
trainlosses = []

testlosses = []
trainaccuracy = []

testaccuracy = []
for e in range(model.epoch):

    trainloss = 0.0

    traintotal = 0

    trainsuccessful = 0

    for traininputs, trainlabels in loadertrain:

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

            for testimages, testlabels in loadertest:

                inputs_, labels_ = testimages.to(device), testlabels.to(device)

                predictions = model(inputs_)

                tloss = criterion(predictions, labels_)

                testloss += tloss.item()

                _, predict = torch.max(predictions.data, 1)

                testtotal += labels_.size(0)

                testsuccessful += (predict == labels_).sum().item()

        trainlosses.append(trainloss/len(loadertrain))

        testlosses.append(testloss/len(loadertest))

        trainaccuracy.append(100*trainsuccessful/traintotal)

        testaccuracy.append(100*testsuccessful/testtotal)

        if e % 5 == 0 or e==39:

            print('Train Accuracy %{:.2f}'.format(100*trainsuccessful/traintotal), end=',  ')

            print('Test Accuracy %{:.2f}'.format(100*testsuccessful/testtotal))
import matplotlib.pyplot as plt

import numpy as np
fig, ((myplt1), (myplt2)) = plt.subplots(2, 1, sharex=True, figsize=(14, 12))

epochaxis = np.arange(0, 40)

myplt1.plot(epochaxis, trainlosses, label='Train Loss', color='purple')

myplt1.plot(epochaxis, testlosses, label='Test Loss', color='blue')

myplt1.fill_between(epochaxis, trainlosses, testlosses, where=(np.array(trainlosses)>np.array(testlosses)), alpha=0.2, color='cyan', label='Train loss > Test Loss')

myplt1.fill_between(epochaxis, trainlosses, testlosses, where=(np.array(trainlosses)<np.array(testlosses)), alpha=0.2, color='orange', label='Test loss > Train Loss')

myplt2.plot(epochaxis, trainaccuracy, label='Train Accuracy %', color='purple')

myplt2.plot(epochaxis, testaccuracy, label='Test Accuracy %', color='blue')

myplt2.fill_between(epochaxis, trainaccuracy, testaccuracy, where=(np.array(trainaccuracy)>np.array(testaccuracy)), interpolate=True, alpha=0.2, color='cyan', label='Train Accuracy > Test Accuracy')

myplt2.fill_between(epochaxis, trainaccuracy, testaccuracy, where=(np.array(testaccuracy)>np.array(trainaccuracy)), interpolate=True, alpha=0.2, color='orange', label='Train Accuracy > Test Accuracy')

myplt2.set_xlabel("Epoch")

myplt1.legend(loc='upper right', frameon=False)

myplt2.legend(loc='lower right', frameon=False)

fig.tight_layout()