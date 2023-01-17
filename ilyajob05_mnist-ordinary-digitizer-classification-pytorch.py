import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lrSheduler
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import pandas as pd
import torch as th
%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import math

# load database 

dataRawTrain = pd.read_csv('../input/train.csv')
dataRawTest = pd.read_csv('../input/test.csv')

dataTrain = np.array(dataRawTrain.iloc[:,1:]).astype('uint8')
dataTrain = dataTrain.reshape(-1, 1, 28, 28)

targetTrain = np.array(dataRawTrain.iloc[:,:1])
zz = np.zeros([len(targetTrain), 10])
for i, d in enumerate(targetTrain):
    zz[i][d.squeeze()] = 0.66
targetTrain = zz

dataTest = np.array(dataRawTest.iloc[:,:]).astype('uint8')
dataTest = dataTest.reshape(-1, 1, 28, 28)

for i in range(1):
    plt.matshow(dataTrain[i][0], cmap='rainbow')
    plt.suptitle('class: {}'.format(targetTrain[i]))

for i in range(1):    
    plt.matshow(dataTest[i][0], cmap='rainbow')
plt.show()

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

dataTrainT = torch.from_numpy(dataTrain).float() / 255.0 - 0.5
targetTrainT = torch.from_numpy(targetTrain).float()

dataTestT = torch.from_numpy(dataTest).float() / 255.0 - 0.5


def processImgTrain(batch):
    outBatchData = []
    outBatchTarget = []
    for i, (img, target) in enumerate(batch):
        outBatchData.append(img)
        outBatchTarget.append(target)
    outTensorData = th.stack(outBatchData)
    outTensorTarget = th.stack(outBatchTarget)
    return [outTensorData, outTensorTarget]

def scaleImgTest(batch):
    outBatchData = []
    outBatchTarget = []
    for i, (img, target) in enumerate(batch):
        outBatchData.append(img)
        outBatchTarget.append(target)
    outTensorData = th.stack(outBatchData)
    outTensorTarget = th.stack(outBatchTarget)
    return [outTensorData, outTensorTarget]

dataTrainArch = TensorDataset(dataTrainT[:-2000], targetTrainT[:-2000])
loaderTrain = DataLoader(dataTrainArch, shuffle=True, batch_size=64, pin_memory=True, collate_fn=processImgTrain)

print('train data size: {}'.format(dataTrainT[:-2000].shape))
print('test data size: {}'.format(dataTrainT[-2000:].shape))

testDBD = dataTestT[-2000:].clone()
testDBT = targetTrainT[-2000:].clone()

dataTestArch = TensorDataset(dataTrainT[-2000:], targetTrainT[-2000:])
loaderTest = DataLoader(dataTestArch, shuffle=True, batch_size=1000, pin_memory=True, collate_fn=scaleImgTest)


class NetMNIST2(nn.Module):
    flag = False

    def __init__(self):
        super(NetMNIST2, self).__init__()

        # First convolution
        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu0 = nn.PReLU()
        self.pool0 = nn.MaxPool2d(2)
        self.drop0 = nn.Dropout2d()

        self.conv1p = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1p = nn.PReLU()
        self.drop1p = nn.Dropout2d()

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d()

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d()

        self.fc1 = nn.Linear(32 * 6 * 6, 10)
        self.fc1Prelu = nn.PReLU()
        self.fc2 = nn.Linear(10, 10)


    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.drop0(x)

        x = self.conv1p(x)
        x = self.relu1p(x)
        x = self.drop1p(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(-1, 128*3*3)
        x = self.fc1(x)
        x = self.fc1Prelu(x)
        x = torch.nn.functional.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    


use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

net = NetMNIST2().to(device)
print(net)
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = F.mse_loss

def trainEpoch(e):
    net.train()
    for i, (data, target) in enumerate(loaderTrain):
        dataCUDA, targetCUDA = Variable(data).to(device), Variable(target.float()).to(device)

        optimizer.zero_grad()
        outModel = net(dataCUDA)

        loss = F.mse_loss(outModel, targetCUDA)
        loss.backward()
        optimizer.step()
    return loss


def testLoss():
    net.eval()
    fit = 0.0

    for i, (data, target) in enumerate(loaderTest):
        dataCUDA, targetCUDA = Variable(data).to(device), Variable(target.float()).to(device)
        outModel = net(dataCUDA)
        pred = outModel.data.max(1)[1]
        targetMaxIndex = targetCUDA.data.max(1)[1]
        fit += pred.eq(targetMaxIndex).cpu().sum()
    acc = float(fit.cpu()) / float(len(loaderTest.dataset))
    return acc

lossProgress = []
for e in range(5):
    lossTrain = trainEpoch(e)
    accTest = testLoss()
    lossProgress.append(accTest)
    print('Test epoch: {}   acc: {}'.format(e, accTest))
    print('Train epoch: {}   loss: {}'.format(e, lossTrain))

plt.plot(lossProgress)


packSize = 10
inputLen = dataTestT.size(0)
indexInput = np.arange(1, inputLen + 1)
result = np.zeros([inputLen],dtype=int)
print(inputLen / packSize)
for i in range(int(inputLen / packSize)):
    dataCUDA = Variable(dataTestT[i * packSize:i*packSize+packSize].cuda()) #add one axis
    outModel = net(dataCUDA)
    result[i * packSize:i*packSize+packSize] = outModel.data.max(1)[1].cpu().numpy().squeeze()

result=pd.DataFrame({'ImageId':indexInput, 'Label':result})
result.to_csv("submission.csv",index=False)
