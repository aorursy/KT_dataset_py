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
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sci
dataRawTrain = pd.read_csv('../input/fashion-mnist_train.csv')
dataRawTest = pd.read_csv('../input/fashion-mnist_test.csv')

dataTrain = np.array(dataRawTrain.iloc[:,1:]).astype('uint8')
dataTrain = dataTrain.reshape(-1, 1, 28, 28)

targetTrain = np.array(dataRawTrain.iloc[:,:1])
zz = np.zeros([len(targetTrain), 10])
for i, d in enumerate(targetTrain):
    zz[i][d.squeeze()] = 0.66
targetTrain = zz

dataTest = np.array(dataRawTest.iloc[:,1:])
dataTest = dataTest.reshape(-1, 1, 28, 28)

targetTest = np.array(dataRawTest.iloc[:,:1])
zz = np.zeros([len(targetTest), 10])
for i, d in enumerate(targetTest):
    zz[i][d.squeeze()] = 0.66
targetTest = zz


for i in range(3):
    plt.matshow(dataTrain[i][0], cmap='rainbow')
    plt.colorbar()
    plt.suptitle('class: {}'.format(targetTrain[i]))
    
for i in range(3):    
    plt.matshow(dataTest[i][0], cmap='rainbow')
    plt.colorbar()
    plt.suptitle('class: {}'.format(targetTest[i]))
plt.show()
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

dataTrainT = torch.from_numpy(dataTrain).float() / np.std(dataTrain) - np.mean(dataTrain)
# dataTrainT = torch.from_numpy(dataTrain).float() / 255.0 - 0.5
targetTrainT = torch.from_numpy(targetTrain).float()
print('train data size: {}'.format(dataTrainT.shape))

dataTestT = torch.from_numpy(dataTest).float() / np.std(dataTest) - np.mean(dataTest)
# dataTestT = torch.from_numpy(dataTest).float() /  255.0 - 0.5
targetTestT = torch.from_numpy(targetTest).float()
print('test data size: {}'.format(dataTestT.shape))

# processing function for trained data
def processImgBatch(batch):
    outBatchData = []
    outBatchTarget = []
    for i, (img, target) in enumerate(batch):
        outBatchData.append(img)
        outBatchTarget.append(target)
    outTensorData = th.stack(outBatchData)
    outTensorTarget = th.stack(outBatchTarget)
    return [outTensorData, outTensorTarget]

dataTrainArch = TensorDataset(dataTrainT, targetTrainT)
loaderTrain = DataLoader(dataTrainArch, shuffle=True, batch_size=64, pin_memory=True, collate_fn=processImgBatch)

dataTestArch = TensorDataset(dataTestT, targetTestT)
loaderTest = DataLoader(dataTestArch, shuffle=False, batch_size=1000, pin_memory=True, collate_fn=processImgBatch)

class NetMNIST2(nn.Module):
    flag = False

    def __init__(self):
        super(NetMNIST2, self).__init__()
        
        # convolution 1
        self.bn0 = nn.BatchNorm2d(1)
        self.conv00 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu00 = nn.PReLU()
        self.conv0 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu0 = nn.PReLU()
        self.pool0 = nn.MaxPool2d(2)
        self.drop0 = nn.Dropout2d()
        
        # convolution 2
        self.bn1 = nn.BatchNorm2d(32)
        self.conv10 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.PReLU()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d()

        # convolution 3
        self.bn2 = nn.BatchNorm2d(64)
        self.conv20 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu20 = nn.PReLU()
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d()

        # full connection layer 1
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(32*6*6, 10)
        self.fc1Prelu = nn.PReLU()
        
        # full connection layer 2
        self.fc2 = nn.Linear(10, 10)
        
    # direct computation
    def forward(self, x):
        x = self.bn0(x)
        x = self.conv00(x)
        x = self.relu00(x)
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.drop0(x)

        x = self.bn1(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.bn2(x)
        x = self.conv20(x)
        x = self.relu20(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
#         show input parameter
#         print(x.shape)
        x = self.bn3(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.fc1Prelu(x)
        x = torch.nn.functional.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

# network initialization
net = NetMNIST2().cuda()
# show network
print(net)
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = F.kl_div

def train(model, optimizer, criterion, dataLoader):
    model.train()
    for i, (data, target) in enumerate(dataLoader):
        dataCUDA, targetCUDA = Variable(data).cuda(), Variable(target.float()).cuda()
        optimizer.zero_grad()
        outModel = model(dataCUDA)
        loss = criterion(outModel, targetCUDA)
        loss.backward()
        optimizer.step()
    return loss


def test(model, dataLoader):
    model.eval()
    fit = 0.0

    for i, (data, target) in enumerate(dataLoader):
        dataCUDA, targetCUDA = Variable(data).cuda(), Variable(target).cuda()
        outModel = model(dataCUDA)
        pred = outModel.data.max(1)[1]
        targetMaxIndex = targetCUDA.data.max(1)[1]     
        fit += pred.eq(targetMaxIndex).cpu().sum()
#     calculation accuracy
    acc = float(fit.cpu().numpy() / float(len(dataLoader.dataset)))
    return acc


lossProgress = []
for e in range(50):
    lossTrain = train(net, optimizer, criterion, loaderTrain)
    accTest = test(net, loaderTest)
    lossProgress.append(accTest)
    print('Epoch: {}   acc: {}   loss: {}'.format(e, accTest, lossTrain))
    
plt.plot(lossProgress)
from sklearn.metrics import confusion_matrix
X = np.empty([0], dtype=np.int16)
Y = np.empty([0], dtype=np.int16)
for i, (data, target) in enumerate(loaderTest):
    dataCUDA, targetCUDA = Variable(data).cuda(), Variable(target).cuda()
    outModel = net(dataCUDA)
    Y = np.append(Y, outModel.data.max(1)[1].cpu().numpy())
    X = np.append(X, targetCUDA.data.max(1)[1].cpu().numpy())
    
from matplotlib.ticker import MultipleLocator
labels = [str(x) for x in range(10)]
confMatrix = confusion_matrix(X.reshape(-1), Y.reshape(-1))

import seaborn as sns
ax= plt.subplot()
sns.heatmap(confMatrix, annot=True, fmt='d', ax = ax, cmap='plasma')

ax.set_xlabel('Predicted')
ax.set_ylabel('Target')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)


